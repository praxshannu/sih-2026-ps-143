"""SENTINEL Intel Service - FastAPI Application.

Intelligence layer that generates LLM narratives, creates legal-grade PDF
case files, and manages evidence integrity via tamper-evident hash chains.
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from loguru import logger

from app.alerter import dispatch_alert
from app.case_file import generate_case_file
from app.hasher import build_custody_chain, compute_evidence_hash, create_custody_entry
from app.narrative import check_llm_availability, generate_narrative, llm_status
from app.schemas import (
    AlertRequest,
    AlertResult,
    CaseFileRequest,
    CaseFileResult,
    HashRequest,
    HashResult,
    HealthResponse,
    NarrativeRequest,
    NarrativeResult,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
CASE_FILE_OUTPUT_DIR = os.getenv("CASE_FILE_OUTPUT_DIR", "/app/data/case_files")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logger.remove()
logger.add(sys.stderr, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")
logger.add(
    "logs/intel_{time:YYYY-MM-DD}.log",
    rotation="1 day",
    retention="30 days",
    level="DEBUG",
)


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    logger.info("SENTINEL Intel Service starting up (provider={})", LLM_PROVIDER)
    Path("logs").mkdir(exist_ok=True)
    Path(CASE_FILE_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    yield
    logger.info("SENTINEL Intel Service shutting down")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SENTINEL Intel Service",
    description=(
        "Intelligence layer for LLM narrative generation, legal-grade PDF "
        "evidence packaging, and tamper-evident chain of custody"
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check with LLM provider status."""
    llm_ok = await check_llm_availability()
    return HealthResponse(
        status="ok" if llm_ok else "degraded",
        llm_provider=LLM_PROVIDER,
        llm_available=llm_ok,
    )


@app.get("/health/llm")
async def llm_health():
    """LLM provider status for the UI badge."""
    return await llm_status()


@app.post("/intel/narrative", response_model=NarrativeResult)
async def create_narrative(req: NarrativeRequest):
    """Generate an LLM case narrative from structured detection data.

    Produces a concise, legally precise incident summary with:
    - 4-sentence max summary
    - Key finding
    - Evidentiary gaps
    - Legal basis (MARPOL citation)
    - Alert priority classification
    """
    t0 = time.monotonic()
    logger.info(
        "[INTEL] Narrative request for case {} (spill={})",
        req.detection.case_id, req.detection.spill_id,
    )

    try:
        # Compute evidence hash if not provided
        evidence_hash = req.evidence_hash
        if not evidence_hash:
            evidence_hash = compute_evidence_hash(
                case_id=req.detection.case_id,
                spill_id=req.detection.spill_id,
                detection_data=req.detection.model_dump(),
                drift_data=req.drift.model_dump() if req.drift else None,
                suspects=[s.model_dump() for s in req.suspects],
            )

        # Generate narrative
        result = await generate_narrative(
            detection=req.detection.model_dump(),
            drift=req.drift.model_dump() if req.drift else None,
            suspects=[s.model_dump() for s in req.suspects],
            evidence_hash=evidence_hash,
        )

        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.info(
            "[INTEL] Narrative generated in {:.1f}ms: priority={}",
            elapsed_ms, result.get("alert_priority"),
        )

        return NarrativeResult(
            summary=result["summary"],
            key_finding=result["key_finding"],
            evidentiary_gaps=result.get("evidentiary_gaps", []),
            legal_basis=result.get("legal_basis", ""),
            alert_priority=result.get("alert_priority", "MEDIUM"),
            evidence_hash=evidence_hash,
            model_used=result.get("model_used", LLM_PROVIDER),
        )

    except Exception as e:
        logger.exception("[INTEL] Narrative generation failed: {}", e)
        raise HTTPException(status_code=500, detail=f"Narrative generation failed: {e}")


@app.post("/intel/case-file", response_model=CaseFileResult)
async def create_case_file(req: CaseFileRequest):
    """Generate a legal-grade PDF evidence case file.

    Produces a complete case file with:
    - SENTINEL letterhead, case number, SHA-256 hash
    - Incident summary (LLM narrative)
    - SAR detection image with spill polygon overlay
    - Geometric properties table
    - Hindcast origin map (95% CI ellipse)
    - 72-hour forecast drift map
    - Suspect vessel ranking table with score breakdown
    - AIS data excerpt (gap period highlighted)
    - Forensic timeline visualization
    - Chain of custody log
    - MARPOL Article 4 citation (if applicable)
    - QR code linking to live dashboard
    """
    t0 = time.monotonic()
    logger.info("[INTEL] Case file request for case {}", req.case_id)

    try:
        # Generate narrative if not provided
        narrative_data = None
        if req.narrative:
            # Pre-generated narrative passed as string; wrap for template
            narrative_data = {"summary": req.narrative}
        else:
            # Generate via LLM
            try:
                evidence_hash = compute_evidence_hash(
                    case_id=req.case_id,
                    spill_id=req.detection.spill_id,
                    detection_data=req.detection.model_dump(),
                    drift_data=req.drift.model_dump() if req.drift else None,
                    suspects=[s.model_dump() for s in req.suspects],
                )
                narrative_data = await generate_narrative(
                    detection=req.detection.model_dump(),
                    drift=req.drift.model_dump() if req.drift else None,
                    suspects=[s.model_dump() for s in req.suspects],
                    evidence_hash=evidence_hash,
                )
            except Exception as e:
                logger.warning("[INTEL] Narrative generation failed, using fallback: {}", e)
                narrative_data = {
                    "summary": f"Oil spill detected at ({req.detection.centroid_lat:.4f}, {req.detection.centroid_lon:.4f}) with {req.detection.confidence * 100:.1f}% confidence.",
                    "key_finding": "Automated detection pending manual review.",
                    "evidentiary_gaps": ["LLM narrative unavailable"],
                    "legal_basis": "",
                    "alert_priority": "MEDIUM",
                }

        # Compute evidence hash
        evidence_hash = compute_evidence_hash(
            case_id=req.case_id,
            spill_id=req.detection.spill_id,
            detection_data=req.detection.model_dump(),
            drift_data=req.drift.model_dump() if req.drift else None,
            suspects=[s.model_dump() for s in req.suspects],
        )

        # Build chain of custody if not provided
        custody_entries: list[dict[str, Any]] = []
        if req.chain_of_custody:
            for c in req.chain_of_custody:
                if hasattr(c, "model_dump"):
                    custody_entries.append(c.model_dump())
                elif isinstance(c, dict):
                    custody_entries.append(c)
        else:
            # Create minimal chain from intel stage
            custody_entries.append(create_custody_entry(
                stage="intel-packaging",
                input_data=req.detection.model_dump(),
                output_data={"evidence_hash": evidence_hash},
                previous_hash="",
            ))

        # Generate PDF
        pdf_path = generate_case_file(
            case_id=req.case_id,
            detection=req.detection.model_dump(),
            drift=req.drift.model_dump() if req.drift else None,
            suspects=[s.model_dump() for s in req.suspects],
            ais_excerpt=req.ais_excerpt.model_dump() if req.ais_excerpt else None,
            narrative=narrative_data,
            chain_of_custody=custody_entries,
            evidence_hash=evidence_hash,
        )

        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.info(
            "[INTEL] Case file generated in {:.1f}ms: {}",
            elapsed_ms, pdf_path,
        )

        return CaseFileResult(
            case_id=req.case_id,
            pdf_path=pdf_path,
            evidence_hash=evidence_hash,
            page_count=0,  # filled by generator
        )

    except Exception as e:
        logger.exception("[INTEL] Case file generation failed: {}", e)
        raise HTTPException(status_code=500, detail=f"Case file generation failed: {e}")


@app.post("/intel/hash", response_model=HashResult)
async def compute_hash(req: HashRequest):
    """Compute SHA-256 evidence chain hash for a pipeline stage.

    Creates a tamper-evident hash chain by linking the current stage
    output to the previous stage hash.
    """
    try:
        output_hash_data = {"case_id": req.case_id, "stage": req.stage, "data": req.data}
        output_hash = compute_evidence_hash(
            case_id=req.case_id,
            spill_id="",
            detection_data=req.data,
        )

        # Chain with previous
        if req.previous_hash:
            import hashlib
            combined = f"{req.previous_hash}{output_hash}".encode("utf-8")
            chain_hash = hashlib.sha256(combined).hexdigest()
        else:
            chain_hash = output_hash

        logger.info(
            "[INTEL] Hash computed for stage '{}': chain={}",
            req.stage, chain_hash[:16],
        )

        return HashResult(
            case_id=req.case_id,
            stage=req.stage,
            input_hash=compute_evidence_hash(case_id=req.case_id, spill_id="", detection_data=req.data),
            output_hash=output_hash,
            chain_hash=chain_hash,
        )

    except Exception as e:
        logger.exception("[INTEL] Hash computation failed: {}", e)
        raise HTTPException(status_code=500, detail=f"Hash computation failed: {e}")


@app.post("/intel/alert", response_model=AlertResult)
async def send_alert(req: AlertRequest):
    """Dispatch case alert via email and/or webhook.

    Sends HTML email notifications and JSON webhook payloads with
    retry logic and failure tracking.
    """
    try:
        result = await dispatch_alert(
            case_id=req.case_id,
            spill_id=req.spill_id,
            alert_priority=req.alert_priority,
            summary=req.summary,
            recipients=req.recipients,
            webhook_urls=req.webhook_urls,
            evidence_hash=req.evidence_hash,
        )

        return AlertResult(
            case_id=req.case_id,
            alert_id=result["alert_id"],
            emails_sent=result["emails_sent"],
            webhooks_sent=result["webhooks_sent"],
            failed_recipients=result["failed_recipients"],
            failed_webhooks=result["failed_webhooks"],
        )

    except Exception as e:
        logger.exception("[INTEL] Alert dispatch failed: {}", e)
        raise HTTPException(status_code=500, detail=f"Alert dispatch failed: {e}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8005,
        reload=os.getenv("ENV") == "development",
    )
