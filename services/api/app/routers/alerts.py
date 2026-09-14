from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from loguru import logger

from app.middleware.auth import get_current_user, require_admin
from app.schemas import (
    AlertCreate,
    AlertListResponse,
    AlertResponse,
    AlertType,
    TokenPayload,
    WebhookTargetCreate,
    WebhookTargetResponse,
)

router = APIRouter(prefix="/alerts", tags=["alerts"])

# In-memory stores — replace with database in production
_alerts: dict[str, dict] = {}
_webhooks: dict[str, dict] = {}
_alert_counter = 0
_webhook_counter = 0


def _next_alert_id() -> str:
    global _alert_counter
    _alert_counter += 1
    return f"alert-{_alert_counter:06d}"


def _next_webhook_id() -> str:
    global _webhook_counter
    _webhook_counter += 1
    return f"wh-{_webhook_counter:06d}"


@router.get("", response_model=AlertListResponse)
async def list_alerts(
    case_id: str | None = Query(None),
    alert_type: AlertType | None = Query(None),
    sent: bool | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    _user: TokenPayload = Depends(get_current_user),
):
    alerts = list(_alerts.values())
    if case_id:
        alerts = [a for a in alerts if a["case_id"] == case_id]
    if alert_type:
        alerts = [a for a in alerts if a["alert_type"] == alert_type]
    if sent is not None:
        alerts = [a for a in alerts if a["sent"] == sent]
    alerts.sort(key=lambda a: a["created_at"], reverse=True)
    total = len(alerts)
    start = (page - 1) * page_size
    end = start + page_size
    return AlertListResponse(
        alerts=[AlertResponse(**a) for a in alerts[start:end]],
        total=total,
    )


@router.post("", response_model=AlertResponse, status_code=status.HTTP_201_CREATED)
async def create_alert(
    body: AlertCreate,
    _user: TokenPayload = Depends(get_current_user),
):
    now = datetime.now(UTC)
    alert_id = _next_alert_id()
    alert = {
        "id": alert_id,
        "case_id": body.case_id,
        "alert_type": body.alert_type,
        "priority": body.priority,
        "title": body.title,
        "message": body.message,
        "metadata": body.metadata,
        "sent": False,
        "sent_at": None,
        "created_at": now,
    }
    _alerts[alert_id] = alert
    logger.info("Alert created: {} ({})", alert_id, body.title)
    return AlertResponse(**alert)


@router.post("/{alert_id}/send", response_model=AlertResponse)
async def mark_alert_sent(
    alert_id: str,
    _user: TokenPayload = Depends(get_current_user),
):
    alert = _alerts.get(alert_id)
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")
    if alert["sent"]:
        raise HTTPException(status_code=400, detail="Alert already sent")
    now = datetime.now(UTC)
    alert["sent"] = True
    alert["sent_at"] = now

    # Dispatch to matching webhooks
    for wh in _webhooks.values():
        if not wh["enabled"]:
            continue
        if alert["alert_type"] in wh.get("alert_types", []) or not wh.get("alert_types"):
            logger.info("Dispatching alert {} to webhook {} ({})", alert_id, wh["id"], wh["url"])

    logger.info("Alert marked as sent: {}", alert_id)
    return AlertResponse(**alert)


@router.get("/webhooks", response_model=list[WebhookTargetResponse])
async def list_webhooks(_user: TokenPayload = Depends(get_current_user)):
    return [WebhookTargetResponse(**wh) for wh in _webhooks.values()]


@router.post("/webhooks", response_model=WebhookTargetResponse, status_code=status.HTTP_201_CREATED)
async def create_webhook(
    body: WebhookTargetCreate,
    _user: TokenPayload = Depends(require_admin),
):
    now = datetime.now(UTC)
    wh_id = _next_webhook_id()
    webhook = {
        "id": wh_id,
        "name": body.name,
        "url": body.url,
        "secret": body.secret,
        "alert_types": body.alert_types,
        "enabled": body.enabled,
        "created_at": now,
    }
    _webhooks[wh_id] = webhook
    logger.info("Webhook created: {} ({})", wh_id, body.name)
    return WebhookTargetResponse(
        id=wh_id,
        name=body.name,
        url=body.url,
        alert_types=body.alert_types,
        enabled=body.enabled,
        created_at=now,
    )


@router.delete("/webhooks/{webhook_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_webhook(
    webhook_id: str,
    _user: TokenPayload = Depends(require_admin),
):
    if webhook_id not in _webhooks:
        raise HTTPException(status_code=404, detail="Webhook not found")
    del _webhooks[webhook_id]
    logger.info("Webhook deleted: {}", webhook_id)
