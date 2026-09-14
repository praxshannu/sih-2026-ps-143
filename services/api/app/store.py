"""Shared pipeline-result cache for the SENTINEL API gateway.

Redis-backed (so the Celery worker and all API replicas share state) with
an in-process fallback when Redis is unreachable. Keys are namespaced per
case: `sentinel:result:{case_id}` -> JSON {stage: payload}.
"""

from __future__ import annotations

import json
import os
from typing import Any

from loguru import logger

REDIS_URL = os.getenv("REDIS_URL", os.getenv("REDIS_BROKER_URL", "redis://localhost:6379/0"))
RESULT_TTL_SECONDS = int(os.getenv("RESULT_TTL_SECONDS", "86400"))

_memory: dict[str, dict[str, Any]] = {}


def _redis():
    try:
        import redis.asyncio as aioredis

        return aioredis.from_url(REDIS_URL, decode_responses=True)
    except Exception as e:
        logger.warning("Redis client unavailable, in-process result cache: {}", e)
        return None


def _key(case_id: str) -> str:
    return f"sentinel:result:{case_id}"


async def save_stage(case_id: str, stage: str, payload: Any) -> None:
    """Persist one pipeline stage result for a case."""
    _memory.setdefault(case_id, {})[stage] = payload
    client = _redis()
    if client is None:
        return
    try:
        raw = await client.get(_key(case_id))
        data = json.loads(raw) if raw else {}
        data[stage] = payload
        await client.setex(_key(case_id), RESULT_TTL_SECONDS, json.dumps(data, default=str))
    except Exception as e:
        logger.warning("Result cache write failed, memory-only: {}", e)
    finally:
        try:
            await client.aclose()
        except Exception:
            pass


async def get_results(case_id: str) -> dict[str, Any]:
    """Return all cached stage results for a case ({} when none)."""
    client = _redis()
    if client is not None:
        try:
            raw = await client.get(_key(case_id))
            if raw:
                return json.loads(raw)
        except Exception as e:
            logger.warning("Result cache read failed, memory fallback: {}", e)
        finally:
            try:
                await client.aclose()
            except Exception:
                pass
    return dict(_memory.get(case_id, {}))


async def get_stage(case_id: str, stage: str) -> Any | None:
    results = await get_results(case_id)
    return results.get(stage)
