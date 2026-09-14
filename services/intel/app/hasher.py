"""SHA-256 chain of custody timestamping for SENTINEL evidence integrity.

Hashes each pipeline stage output and creates a tamper-evident chain.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any


def compute_sha256(data: bytes) -> str:
    """Compute SHA-256 hex digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def compute_object_hash(obj: Any) -> str:
    """Deterministically hash a JSON-serializable object.

    Uses sorted keys and no whitespace for reproducibility.
    """
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return compute_sha256(canonical.encode("utf-8"))


def chain_hash(previous_hash: str, current_hash: str) -> str:
    """Compute chained hash: SHA-256(previous || current).

    This creates a tamper-evident chain where modifying any stage
    invalidates all subsequent hashes.
    """
    combined = f"{previous_hash}{current_hash}".encode()
    return compute_sha256(combined)


def create_custody_entry(
    stage: str,
    input_data: Any,
    output_data: Any,
    previous_hash: str = "",
    service: str = "sentinel-intel",
) -> dict[str, Any]:
    """Create a single chain of custody entry.

    Returns dict with stage, timestamp, input_hash, output_hash, chain_hash.
    """
    input_hash = compute_object_hash(input_data)
    output_hash = compute_object_hash(output_data)
    chained = chain_hash(previous_hash, output_hash) if previous_hash else output_hash

    return {
        "stage": stage,
        "timestamp": datetime.now(UTC).isoformat(),
        "input_hash": input_hash,
        "output_hash": output_hash,
        "chain_hash": chained,
        "service": service,
    }


def build_custody_chain(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build a complete chain of custody from a list of stage outputs.

    Each entry is chained to the previous one via SHA-256.
    Returns the full chain with cumulative hashes.
    """
    chain: list[dict[str, Any]] = []
    previous_hash = ""

    for entry in entries:
        stage = entry.get("stage", "unknown")
        input_data = entry.get("input_data", {})
        output_data = entry.get("output_data", {})
        service = entry.get("service", "sentinel-intel")

        custody = create_custody_entry(
            stage=stage,
            input_data=input_data,
            output_data=output_data,
            previous_hash=previous_hash,
            service=service,
        )
        chain.append(custody)
        previous_hash = custody["chain_hash"]

    return chain


def verify_chain(chain: list[dict[str, Any]]) -> tuple[bool, int]:
    """Verify the integrity of a chain of custody.

    Returns (is_valid, failed_index) where failed_index is -1 if all valid.
    """
    previous_hash = ""

    for i, entry in enumerate(chain):
        output_hash = compute_object_hash(
            {"stage": entry["stage"], "output": entry.get("output_hash", "")}
        )

        if entry.get("input_hash", "") == "":
            # Cannot verify input hash without original data
            pass

        expected_chain = (
            chain_hash(previous_hash, entry["output_hash"])
            if previous_hash
            else entry["output_hash"]
        )

        if entry.get("chain_hash", "") != expected_chain:
            return False, i

        previous_hash = entry["chain_hash"]

    return True, -1


def compute_evidence_hash(
    case_id: str,
    spill_id: str,
    detection_data: dict[str, Any],
    drift_data: dict[str, Any] | None = None,
    suspects: list[dict[str, Any]] | None = None,
) -> str:
    """Compute a single evidence package hash from all case components.

    Canonical coverage (sorted keys): detection confidence, drift origin
    coords + ellipse, and the top-3 suspect scores including SHAP breakdowns.
    This is the root hash embedded in the PDF and QR code.
    """
    package = canonical_evidence_package(
        case_id=case_id,
        spill_id=spill_id,
        detection_data=detection_data,
        drift_data=drift_data,
        suspects=suspects,
    )
    return compute_object_hash(package)


def canonical_evidence_package(
    case_id: str,
    spill_id: str,
    detection_data: dict[str, Any],
    drift_data: dict[str, Any] | None = None,
    suspects: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the canonical, alphabetically-sorted evidence package."""
    d = detection_data or {}
    detection = {
        "area_m2": d.get("area_m2"),
        "case_id": d.get("case_id", case_id),
        "centroid_lat": d.get("centroid_lat"),
        "centroid_lon": d.get("centroid_lon"),
        "confidence": d.get("confidence"),
        "detected_at": str(d.get("detected_at", "")),
        "spill_id": d.get("spill_id", spill_id),
    }
    package: dict[str, Any] = {
        "case_id": case_id,
        "detection": detection,
        "spill_id": spill_id,
    }
    if drift_data:
        dr = drift_data
        ell = dr.get("origin_ellipse") or {}
        package["drift"] = {
            "forcing_source": dr.get("forcing_source"),
            "origin_lat": ell.get("center_lat", dr.get("origin_lat")),
            "origin_lon": ell.get("center_lon", dr.get("origin_lon")),
            "semi_major_km": ell.get("semi_major_km"),
            "semi_minor_km": ell.get("semi_minor_km"),
            "xgb_correction_m": dr.get("xgb_correction_m"),
            "xgb_residual_applied": dr.get("xgb_residual_applied"),
        }
    top3 = []
    for s in (suspects or [])[:3]:
        top3.append(
            {
                "composite_score": s.get("composite_score"),
                "confidence_lower": s.get("confidence_lower"),
                "confidence_upper": s.get("confidence_upper"),
                "fuzzy_score": s.get("fuzzy_score"),
                "mmsi": s.get("mmsi"),
                "shap_breakdown": s.get("shap_breakdown", {}),
                "vessel_name": s.get("vessel_name"),
                "xgb_method": s.get("xgb_method"),
                "xgb_score": s.get("xgb_score"),
            }
        )
    package["suspects_top3"] = top3
    # Alphabetical key order at every level for canonical JSON.
    return json.loads(json.dumps(package, sort_keys=True, default=str))
