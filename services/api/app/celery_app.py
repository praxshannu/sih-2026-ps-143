from __future__ import annotations

import os

from celery import Celery

REDIS_BROKER = os.getenv("REDIS_BROKER_URL", "redis://localhost:6379/0")
REDIS_BACKEND = os.getenv("REDIS_RESULT_BACKEND", REDIS_BROKER)

celery_app = Celery(
    "sentinel_pipeline",
    broker=REDIS_BROKER,
    backend=REDIS_BACKEND,
    include=["app.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_soft_time_limit=300,
    task_time_limit=600,
    result_expires=3600,
    broker_connection_retry_on_startup=True,
)

# Pipeline task routing
celery_app.conf.task_routes = {
    "app.tasks.run_detect_pipeline": {"queue": "detect"},
    "app.tasks.run_drift_pipeline": {"queue": "drift"},
    "app.tasks.run_attribute_pipeline": {"queue": "attribute"},
    "app.tasks.run_intel_pipeline": {"queue": "intel"},
    "app.tasks.run_full_pipeline": {"queue": "pipeline"},
    "app.tasks.generate_case_file": {"queue": "pipeline"},
}
