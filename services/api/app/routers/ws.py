"""Redis-backed WebSocket hub for live pipeline events.

Local subscriber sets are kept per process for fan-out, but every broadcast
is also published to Redis (`sentinel:ws:{case_id}`) so events survive
multiple API replicas. The Celery worker publishes stage events straight to
Redis; each API replica forwards them to its local sockets. Falls back to
in-process-only when Redis is unreachable.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections import defaultdict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from loguru import logger

from app.middleware.auth import _decode_token
from app.schemas import WSEvent, WSEventType

router = APIRouter(tags=["websocket"])

REDIS_URL = os.getenv("REDIS_URL", os.getenv("REDIS_BROKER_URL", "redis://localhost:6379/0"))


def _channel(case_id: str) -> str:
    return f"sentinel:ws:{case_id}"


class ConnectionManager:
    """Local sockets + Redis bus fan-out grouped by case_id."""

    def __init__(self) -> None:
        self._connections: dict[str, set[WebSocket]] = defaultdict(set)
        self._lock = asyncio.Lock()
        self._listeners: dict[str, asyncio.Task] = {}

    async def connect(self, ws: WebSocket, case_id: str) -> None:
        await ws.accept()
        async with self._lock:
            self._connections[case_id].add(ws)
            start_listener = case_id not in self._listeners
        if start_listener:
            self._listeners[case_id] = asyncio.create_task(self._redis_listener(case_id))
        logger.info("WS connected: case={} total={}", case_id, len(self._connections[case_id]))

    async def disconnect(self, ws: WebSocket, case_id: str) -> None:
        async with self._lock:
            self._connections[case_id].discard(ws)
            empty = not self._connections[case_id]
            if empty:
                self._connections.pop(case_id, None)
        if empty:
            task = self._listeners.pop(case_id, None)
            if task is not None:
                task.cancel()
        logger.info("WS disconnected: case={}", case_id)

    async def _send_local(self, case_id: str, message: str) -> None:
        dead: list[WebSocket] = []
        async with self._lock:
            targets = list(self._connections.get(case_id, set()))
        for ws in targets:
            try:
                await ws.send_text(message)
            except Exception:
                dead.append(ws)
        if dead:
            async with self._lock:
                for ws in dead:
                    self._connections[case_id].discard(ws)

    async def _redis_listener(self, case_id: str) -> None:
        try:
            import redis.asyncio as aioredis

            client = aioredis.from_url(REDIS_URL, decode_responses=True)
            pubsub = client.pubsub()
            await pubsub.subscribe(_channel(case_id))
            try:
                async for msg in pubsub.listen():
                    if msg.get("type") != "message":
                        continue
                    await self._send_local(case_id, str(msg.get("data", "")))
            finally:
                try:
                    await pubsub.unsubscribe(_channel(case_id))
                    await client.aclose()
                except Exception:
                    pass
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning("WS Redis listener unavailable for {}: {}", case_id, e)

    async def broadcast(self, case_id: str, event: WSEvent) -> None:
        message = event.model_dump_json()
        # Redis first (cross-replica), then local (same-process immediacy).
        try:
            import redis.asyncio as aioredis

            client = aioredis.from_url(REDIS_URL, decode_responses=True)
            try:
                await client.publish(_channel(case_id), message)
            finally:
                try:
                    await client.aclose()
                except Exception:
                    pass
        except Exception as e:
            logger.warning("WS Redis publish failed, local-only broadcast: {}", e)
        await self._send_local(case_id, message)

    async def broadcast_all(self, event: WSEvent) -> None:
        for case_id in list(self._connections.keys()):
            await self.broadcast(case_id, event)

    @property
    def total_connections(self) -> int:
        return sum(len(v) for v in self._connections.values())


manager = ConnectionManager()


def get_ws_manager() -> ConnectionManager:
    return manager


async def _serve(ws: WebSocket, case_id: str) -> None:
    await manager.connect(ws, case_id)
    try:
        while True:
            data = await ws.receive_text()
            try:
                msg = json.loads(data)
                mtype = msg.get("type") or msg.get("action")
                if mtype == "subscribe":
                    new_case = msg.get("case_id") or msg.get("channel", "")
                    if new_case and new_case != case_id:
                        await manager.disconnect(ws, case_id)
                        case_id = new_case
                        await manager.connect(ws, case_id)
                    await ws.send_text(json.dumps({"type": "subscribed", "case_id": case_id}))
                elif mtype in ("ping", "heartbeat"):
                    await ws.send_text(json.dumps({"type": "pong"}))
            except json.JSONDecodeError:
                await ws.send_text(json.dumps({"type": "error", "detail": "Invalid JSON"}))
    except WebSocketDisconnect:
        pass
    finally:
        await manager.disconnect(ws, case_id)


@router.websocket("/ws")
async def websocket_endpoint(
    ws: WebSocket,
    token: str = Query(default=""),
    case_id: str = Query(default=""),
):
    if token:
        try:
            _decode_token(token)
        except Exception:
            await ws.close(code=4001, reason="Invalid token")
            return

    if not case_id:
        await ws.close(code=4000, reason="case_id query parameter required")
        return

    await _serve(ws, case_id)


@router.websocket("/ws/cases/{case_id}")
async def websocket_case_endpoint(ws: WebSocket, case_id: str):
    """Spec-shaped per-case socket: /api/v1/ws/cases/:caseId (token optional)."""
    await _serve(ws, case_id)


async def broadcast_event(event: WSEvent) -> None:
    await manager.broadcast(event.payload.get("case_id", ""), event)
