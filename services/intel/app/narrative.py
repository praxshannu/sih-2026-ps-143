"""LLM narrative generator for SENTINEL case summaries.

Supports OpenAI GPT-4o (primary) and Ollama (offline fallback).
Configured via LLM_PROVIDER env var (openai|ollama).
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
from loguru import logger

SYSTEM_PROMPT = """You are SENTINEL, an autonomous maritime crime intelligence analyst for India's National Technical Research Organisation (NTRO). Generate concise, factual, legally precise incident summaries from structured detection data.

Rules:
- Write in third-person declarative style
- Use UTC timestamps always
- Express confidence quantitatively (e.g., "91.4% confidence")
- Flag any evidentiary gaps explicitly
- Never speculate beyond the data
- Maximum 4 sentences for main summary
- End with: "Evidence package hash: {sha256}"
"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string", "description": "Main incident summary, max 4 sentences"},
        "key_finding": {"type": "string", "description": "Single most important finding"},
        "evidentiary_gaps": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of evidentiary gaps identified",
        },
        "legal_basis": {"type": "string", "description": "Applicable legal framework citation"},
        "alert_priority": {
            "type": "string",
            "enum": ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        },
    },
    "required": ["summary", "key_finding", "evidentiary_gaps", "legal_basis", "alert_priority"],
}

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "60"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))


def _build_user_prompt(
    detection: dict[str, Any],
    drift: dict[str, Any] | None,
    suspects: list[dict[str, Any]],
    evidence_hash: str,
) -> str:
    """Construct the user prompt from structured detection data."""
    parts: list[str] = []

    # Detection summary
    d = detection
    parts.append(
        f"DETECTION EVENT:\n"
        f"  Spill ID: {d['spill_id']}\n"
        f"  Case ID: {d['case_id']}\n"
        f"  Location: ({d['centroid_lat']:.6f}, {d['centroid_lon']:.6f})\n"
        f"  Area: {d['area_m2']:.1f} m²\n"
        f"  Confidence: {d['confidence'] * 100:.1f}%\n"
        f"  Estimated age: {d['age_hours']:.1f} hours\n"
        f"  Detected (UTC): {d['detected_at']}"
    )

    # Drift data
    if drift:
        parts.append(
            f"\nDRIFT ANALYSIS:\n"
            f"  Probable origin: ({drift['origin_lat']:.6f}, {drift['origin_lon']:.6f})\n"
            f"  Origin uncertainty: ±{drift.get('origin_lat_sigma', 0):.4f}° lat, "
            f"±{drift.get('origin_lon_sigma', 0):.4f}° lon\n"
            f"  95% CI ellipse: {drift.get('semi_major_km', 0):.1f} × "
            f"{drift.get('semi_minor_km', 0):.1f} km\n"
            f"  Regime: {drift.get('regime', 'unknown')}"
        )

    # Suspects
    if suspects:
        suspect_lines = []
        for s in suspects[:5]:
            suspect_lines.append(
                f"  #{s['rank']} {s['vessel_name']} (MMSI {s['mmsi']}): "
                f"score={s['composite_score']:.3f}, "
                f"gap={s.get('ais_gap_minutes', 0):.0f}min"
                f"{', DARK VESSEL' if s.get('is_dark_vessel') else ''}"
            )
        parts.append(f"\nSUSPECT VESSELS (top {len(suspects)}):\n" + "\n".join(suspect_lines))

    # Evidence hash
    parts.append(f"\nEVIDENCE PACKAGE HASH: {evidence_hash}")

    return "\n".join(parts)


async def _call_openai(user_prompt: str, evidence_hash: str) -> dict[str, Any]:
    """Call OpenAI GPT-4o with structured output."""
    system = SYSTEM_PROMPT.replace("{sha256}", evidence_hash)

    async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": OPENAI_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.1,
                "max_tokens": 800,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "sentinel_narrative",
                        "strict": True,
                        "schema": RESPONSE_SCHEMA,
                    },
                },
            },
        )
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        return json.loads(content)


async def _call_ollama(user_prompt: str, evidence_hash: str) -> dict[str, Any]:
    """Call local Ollama instance for offline narrative generation."""
    system = SYSTEM_PROMPT.replace("{sha256}", evidence_hash)

    async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
        response = await client.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.1, "num_predict": 800},
            },
        )
        response.raise_for_status()
        data = response.json()
        content = data["message"]["content"]
        parsed = json.loads(content)

        # Ollama may not enforce schema strictly; fill missing fields
        parsed.setdefault("summary", "")
        parsed.setdefault("key_finding", "")
        parsed.setdefault("evidentiary_gaps", [])
        parsed.setdefault("legal_basis", "")
        parsed.setdefault("alert_priority", "MEDIUM")
        return parsed


async def generate_narrative(
    detection: dict[str, Any],
    drift: dict[str, Any] | None = None,
    suspects: list[dict[str, Any]] | None = None,
    evidence_hash: str = "",
) -> dict[str, Any]:
    """Generate an LLM case narrative from structured detection data.

    Dispatches to OpenAI or Ollama based on LLM_PROVIDER env var, with a
    retry-with-fix loop (up to LLM_MAX_RETRIES): each schema validation
    failure is sent back as a follow-up message. Falls back to the offline
    template so the pipeline never blocks on LLM availability.
    Returns dict with keys: summary, key_finding, evidentiary_gaps,
    legal_basis, alert_priority (+ evidence_hash, model_used).
    """
    user_prompt = _build_user_prompt(detection, drift, suspects or [], evidence_hash)
    logger.info("Generating narrative via {} for case {}", LLM_PROVIDER, detection.get("case_id"))

    last_error = ""
    for attempt in range(1, LLM_MAX_RETRIES + 1):
        try:
            prompt = (
                user_prompt
                if attempt == 1
                else (
                    f"{user_prompt}\n\nPREVIOUS OUTPUT FAILED VALIDATION: {last_error}\n"
                    "Return ONLY valid JSON matching the required schema."
                )
            )
            if LLM_PROVIDER == "ollama":
                result = await _call_ollama(prompt, evidence_hash)
            else:
                result = await _call_openai(prompt, evidence_hash)
            _validate_narrative(result)
            result["evidence_hash"] = evidence_hash
            result["model_used"] = OPENAI_MODEL if LLM_PROVIDER == "openai" else OLLAMA_MODEL
            logger.info(
                "Narrative generated: priority={}, gaps={}",
                result.get("alert_priority"),
                len(result.get("evidentiary_gaps", [])),
            )
            return result
        except Exception as e:
            last_error = str(e)[:500]
            logger.warning(
                "Narrative attempt {}/{} failed: {}", attempt, LLM_MAX_RETRIES, last_error
            )

    logger.warning("All LLM attempts failed, using offline template")
    result = offline_narrative(detection, drift, suspects or [], evidence_hash)
    result["evidence_hash"] = evidence_hash
    return result


def _validate_narrative(result: dict[str, Any]) -> None:
    """Validate LLM output against RESPONSE_SCHEMA; raises ValueError."""
    for key in ("summary", "key_finding", "evidentiary_gaps", "legal_basis", "alert_priority"):
        if key not in result:
            raise ValueError(f"missing required field: {key}")
    if result["alert_priority"] not in ("LOW", "MEDIUM", "HIGH", "CRITICAL"):
        raise ValueError(f"invalid alert_priority: {result.get('alert_priority')}")
    if not isinstance(result["evidentiary_gaps"], list):
        raise ValueError("evidentiary_gaps must be a list")


def offline_narrative(
    detection: dict[str, Any],
    drift: dict[str, Any] | None,
    suspects: list[dict[str, Any]],
    evidence_hash: str,
) -> dict[str, Any]:
    """Deterministic offline template (no LLM needed)."""
    d = detection or {}
    conf = float(d.get("confidence", 0.0)) * 100.0
    lat = float(d.get("centroid_lat", 0.0))
    lon = float(d.get("centroid_lon", 0.0))
    top = suspects[0] if suspects else {}
    gaps: list[str] = []
    if not drift:
        gaps.append("Drift hindcast unavailable")
    if not suspects:
        gaps.append("No suspect vessels ranked")
    if conf < 50:
        gaps.append("Low detection confidence; recommend SAR re-tasking")
    summary = (
        f"Oil spill detected at ({lat:.4f}, {lon:.4f}) with {conf:.1f}% confidence. "
        f"Top suspect: {top.get('vessel_name', 'UNKNOWN')} "
        f"(MMSI {top.get('mmsi', 'UNKNOWN')}). "
        f"Evidence package hash: {evidence_hash}"
    )
    return {
        "summary": summary,
        "key_finding": (
            f"Primary suspect {top.get('vessel_name', 'UNKNOWN')} scores "
            f"{float(top.get('composite_score', 0.0)):.3f}."
            if top
            else "No suspects ranked."
        ),
        "evidentiary_gaps": gaps,
        "legal_basis": "MARPOL Annex I — discharge of oil prohibited; flag/port state jurisdiction applies.",
        "alert_priority": "HIGH" if conf >= 75 else ("MEDIUM" if conf >= 50 else "LOW"),
        "model_used": "offline_template",
    }


async def llm_status() -> dict[str, Any]:
    """Provider status for the UI badge: {provider, model, available, latency_ms}."""
    import time as _time

    provider = LLM_PROVIDER
    model = OPENAI_MODEL if provider == "openai" else OLLAMA_MODEL
    start = _time.monotonic()
    available = await check_llm_availability()
    latency_ms = round((_time.monotonic() - start) * 1000.0, 1)
    return {
        "provider": provider,
        "model": model,
        "available": bool(available),
        "latency_ms": latency_ms,
    }


async def check_llm_availability() -> bool:
    """Check whether the configured LLM provider is reachable.

    For Ollama this means the server is up AND the configured model is
    pulled — otherwise the UI badge would claim LIVE with no model.
    """
    try:
        if LLM_PROVIDER == "ollama":
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
                if resp.status_code != 200:
                    return False
                try:
                    names = [str(m.get("name", "")) for m in resp.json().get("models", [])]
                except Exception:
                    return False
                want = OLLAMA_MODEL.strip()
                base = want.split(":")[0]
                return any(n == want or n.startswith(base + ":") for n in names)
        else:
            return bool(OPENAI_API_KEY)
    except Exception:
        return False
