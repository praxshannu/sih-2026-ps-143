"""Email and webhook alert dispatcher for SENTINEL case notifications.

Dispatches alerts via SMTP email and HTTP webhooks with retry logic.
"""

from __future__ import annotations

import os
import smtplib
import uuid
from datetime import UTC, datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import httpx
from loguru import logger

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.sentinel.gov.in")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_FROM = os.getenv("SMTP_FROM", "sentinel-noreply@sentinel.gov.in")
ALERT_WEBHOOK_TIMEOUT = float(os.getenv("ALERT_WEBHOOK_TIMEOUT", "15"))
ALERT_RETRY_COUNT = int(os.getenv("ALERT_RETRY_COUNT", "3"))


def _build_email_html(
    case_id: str,
    spill_id: str,
    summary: str,
    alert_priority: str,
    evidence_hash: str,
) -> str:
    """Build HTML email body for case alert."""
    priority_colors = {
        "LOW": "#2ecc71",
        "MEDIUM": "#f39c12",
        "HIGH": "#e74c3c",
        "CRITICAL": "#8e44ad",
    }
    color = priority_colors.get(alert_priority, "#95a5a6")

    return (
        """
    <!DOCTYPE html>
    <html>
    <head><meta charset="utf-8"></head>
    <body style="font-family: 'Segoe UI', Arial, sans-serif; margin: 0; padding:"""
        """ 20px; background: #f4f4f4;">
      <div style="max-width: 640px; margin: 0 auto; background: #fff; border-radius:"""
        """ 8px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
        <div style="background: #1a1a2e; color: #fff; padding: 20px 24px;">
          <h2 style="margin: 0; font-size: 18px;">SENTINEL Maritime Intelligence</h2>
          <p style="margin: 4px 0 0; font-size: 12px; color: #aaa;">NTRO - National"""
        f""" Technical Research Organisation</p>
        </div>
        <div style="padding: 24px;">
          <div style="display: inline-block; background: {color}; color: #fff;"""
        """ padding: 4px 12px; border-radius: 4px; font-size: 12px; font-weight: bold;"""
        f""" margin-bottom: 16px;">
            {alert_priority} PRIORITY
          </div>
          <table style="width: 100%; font-size: 14px; margin-bottom: 16px;">
            <tr><td style="padding: 6px 0; color: #666;">Case ID</td><td"""
        f""" style="padding: 6px 0; font-family: monospace;">{case_id}</td></tr>
            <tr><td style="padding: 6px 0; color: #666;">Spill ID</td><td"""
        f""" style="padding: 6px 0; font-family: monospace;">{spill_id}</td></tr>
            <tr><td style="padding: 6px 0; color: #666;">Timestamp (UTC)</td><td"""
        """ style="padding: 6px"""
        f""" 0;">{datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")}</td></tr>
            <tr><td style="padding: 6px 0; color: #666;">Evidence Hash</td><td"""
        """ style="padding: 6px 0; font-family: monospace; font-size: 11px; word-break:"""
        f""" break-all;">{evidence_hash}</td></tr>
          </table>
          <div style="background: #f8f9fa; border-left: 4px solid {color}; padding:"""
        f""" 16px; margin-bottom: 16px; font-size: 14px; line-height: 1.6;">
            {summary}
          </div>
          <p style="font-size: 12px; color: #999; margin: 0;">
            This is an automated alert from SENTINEL. Case file and evidence package"""
        """ are available on the dashboard.
          </p>
        </div>
      </div>
    </body>
    </html>
    """
    )


async def _send_email(
    recipient: str,
    subject: str,
    html_body: str,
) -> bool:
    """Send alert email via SMTP. Returns True on success."""
    if not SMTP_USER or not SMTP_PASSWORD:
        logger.warning("SMTP credentials not configured, skipping email to {}", recipient)
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = SMTP_FROM
    msg["To"] = recipient
    msg.attach(MIMEText(html_body, "html"))

    try:
        loop = __import__("asyncio").get_event_loop()
        await loop.run_in_executor(
            None,
            _smtp_send,
            msg,
        )
        logger.info("Alert email sent to {}", recipient)
        return True
    except Exception as e:
        logger.error("Failed to send email to {}: {}", recipient, e)
        return False


def _smtp_send(msg: MIMEMultipart) -> None:
    """Synchronous SMTP send (runs in thread pool)."""
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.send_message(msg)


async def _send_webhook(
    url: str,
    payload: dict,
) -> bool:
    """Send alert payload to a webhook endpoint. Returns True on success."""
    headers = {
        "Content-Type": "application/json",
        "X-SENTINEL-Alert": "true",
        "User-Agent": "SENTINEL-Intel/1.0",
    }

    for attempt in range(1, ALERT_RETRY_COUNT + 1):
        try:
            async with httpx.AsyncClient(timeout=ALERT_WEBHOOK_TIMEOUT) as client:
                response = await client.post(url, json=payload, headers=headers)
                if response.status_code < 300:
                    logger.info("Webhook delivered to {} (attempt {})", url, attempt)
                    return True
                logger.warning(
                    "Webhook {} returned {} (attempt {}/{})",
                    url,
                    response.status_code,
                    attempt,
                    ALERT_RETRY_COUNT,
                )
        except Exception as e:
            logger.warning(
                "Webhook {} failed (attempt {}/{}): {}", url, attempt, ALERT_RETRY_COUNT, e
            )

    return False


async def dispatch_alert(
    case_id: str,
    spill_id: str,
    alert_priority: str,
    summary: str,
    recipients: list[str],
    webhook_urls: list[str],
    evidence_hash: str = "",
) -> dict:
    """Dispatch case alert via email and webhooks.

    Returns dispatch result with counts and failures.
    """
    alert_id = f"ALT-{uuid.uuid4().hex[:12].upper()}"
    subject = f"[SENTINEL {alert_priority}] Case {case_id} - Oil Spill Alert"

    logger.info(
        "Dispatching alert {} for case {} (priority={}, emails={}, webhooks={})",
        alert_id,
        case_id,
        alert_priority,
        len(recipients),
        len(webhook_urls),
    )

    # Send emails
    html_body = _build_email_html(case_id, spill_id, summary, alert_priority, evidence_hash)
    emails_sent = 0
    failed_recipients: list[str] = []

    for recipient in recipients:
        if await _send_email(recipient, subject, html_body):
            emails_sent += 1
        else:
            failed_recipients.append(recipient)

    # Send webhooks
    webhook_payload = {
        "alert_id": alert_id,
        "case_id": case_id,
        "spill_id": spill_id,
        "alert_priority": alert_priority,
        "summary": summary,
        "evidence_hash": evidence_hash,
        "timestamp": datetime.now(UTC).isoformat(),
        "source": "sentinel-intel",
    }
    webhooks_sent = 0
    failed_webhooks: list[str] = []

    for url in webhook_urls:
        if await _send_webhook(url, webhook_payload):
            webhooks_sent += 1
        else:
            failed_webhooks.append(url)

    logger.info(
        "Alert {} dispatched: {}/{} emails, {}/{} webhooks",
        alert_id,
        emails_sent,
        len(recipients),
        webhooks_sent,
        len(webhook_urls),
    )

    return {
        "alert_id": alert_id,
        "emails_sent": emails_sent,
        "webhooks_sent": webhooks_sent,
        "failed_recipients": failed_recipients,
        "failed_webhooks": failed_webhooks,
    }
