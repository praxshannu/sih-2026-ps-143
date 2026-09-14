"""APScheduler-based orbital watch daemon for the SENTINEL ingest service.

Maintains a schedule of Sentinel-1 passes over Indian EEZ and triggers
data fetch every 6 hours (or on-demand). Coordinates fetch of all data sources.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Coroutine
from datetime import UTC, datetime
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger

from .models.schemas import (
    DataSource,
    JobStatus,
    SchedulerStatus,
    SourceStatus,
)


class IngestScheduler:
    """APScheduler-based daemon that orchestrates data ingestion.

    Maintains:
      - Sentinel-1 orbital schedule awareness
      - 6-hour interval triggers for all data sources
      - Per-source status tracking
      - Manual on-demand trigger support
    """

    def __init__(self, interval_hours: int = 6) -> None:
        self.scheduler = AsyncIOScheduler(timezone="UTC")
        self.interval_hours = interval_hours
        self.start_time = time.time()
        self._running = False

        # Per-source status tracking
        self._source_status: dict[DataSource, SourceStatus] = {
            ds: SourceStatus(source=ds) for ds in DataSource
        }

        # Callbacks for actual fetch operations (set by the FastAPI app)
        self._fetch_callbacks: dict[DataSource, Callable[..., Coroutine[Any, Any, Any]]] = {}

        # Sentinel-1 orbital schedule (simplified)
        self._orbital_schedule: list[dict] = []

    def register_fetch_callback(
        self,
        source: DataSource,
        callback: Callable[..., Coroutine[Any, Any, Any]],
    ) -> None:
        """Register an async callback for a data source fetch."""
        self._fetch_callbacks[source] = callback
        logger.info("Registered fetch callback for {}", source.value)

    def start(self) -> None:
        """Start the scheduler daemon."""
        if self._running:
            logger.warning("Scheduler already running")
            return

        # Add the main ingestion job
        self.scheduler.add_job(
            self._run_full_cycle,
            trigger=IntervalTrigger(hours=self.interval_hours),
            id="full_ingest_cycle",
            name="Full Ingestion Cycle",
            replace_existing=True,
            next_run_time=datetime.now(UTC),  # run immediately on start
        )

        # Add Sentinel-1 orbital check (every 30 minutes to check for new passes)
        self.scheduler.add_job(
            self._check_orbital_schedule,
            trigger=IntervalTrigger(minutes=30),
            id="orbital_check",
            name="Orbital Schedule Check",
            replace_existing=True,
        )

        self.scheduler.start()
        self._running = True
        self.start_time = time.time()
        logger.info(
            "Ingest scheduler started: interval={}h, jobs={}",
            self.interval_hours,
            [j.id for j in self.scheduler.get_jobs()],
        )

    def stop(self) -> None:
        """Stop the scheduler daemon."""
        if self._running:
            self.scheduler.shutdown(wait=False)
            self._running = False
            logger.info("Ingest scheduler stopped")

    def get_status(self) -> SchedulerStatus:
        """Get current scheduler status and last fetch times."""
        next_run = None
        jobs = self.scheduler.get_jobs()
        if jobs:
            next_times = [j.next_run_time for j in jobs if j.next_run_time]
            if next_times:
                next_run = min(next_times)

        return SchedulerStatus(
            running=self._running,
            next_run_time=next_run,
            interval_hours=self.interval_hours,
            sources=list(self._source_status.values()),
            uptime_seconds=round(time.time() - self.start_time, 1),
        )

    async def trigger_source(
        self,
        source: DataSource,
        params: dict | None = None,
    ) -> Any:
        """Manually trigger a specific data source fetch.

        Parameters
        ----------
        source : Which data source to fetch.
        params : Optional parameters for the fetch.

        Returns
        -------
        The result from the fetch callback.
        """
        logger.info("Manual trigger for source: {}", source.value)

        self._source_status[source].last_fetch_status = JobStatus.RUNNING
        self._source_status[source].last_fetch_time = datetime.now(UTC)

        t0 = time.time()
        try:
            callback = self._fetch_callbacks.get(source)
            if callback is None:
                raise ValueError(f"No fetch callback registered for {source.value}")

            result = await callback(**(params or {}))

            self._source_status[source].last_fetch_status = JobStatus.COMPLETED
            self._source_status[source].last_fetch_duration_seconds = round(time.time() - t0, 2)
            self._source_status[source].total_fetches += 1

            logger.info("Source {} fetch completed in {:.1f}s", source.value, time.time() - t0)
            return result

        except Exception as e:
            self._source_status[source].last_fetch_status = JobStatus.FAILED
            self._source_status[source].last_error = str(e)
            self._source_status[source].last_fetch_duration_seconds = round(time.time() - t0, 2)
            logger.error("Source {} fetch failed: {}", source.value, str(e))
            raise

    async def trigger_full_cycle(
        self,
        sources: list[DataSource] | None = None,
        force: bool = False,
    ) -> dict[DataSource, Any]:
        """Trigger a full ingestion cycle (all or selected sources).

        Returns dict of source -> result.
        """
        logger.info("Full ingestion cycle triggered (force={})", force)
        targets = sources or list(DataSource)
        results: dict[DataSource, Any] = {}

        for source in targets:
            try:
                result = await self.trigger_source(source)
                results[source] = result
            except Exception as e:
                results[source] = {"error": str(e)}

        logger.info("Full ingestion cycle complete: {} sources processed", len(results))
        return results

    async def _run_full_cycle(self) -> None:
        """Scheduled job: run full ingestion cycle."""
        logger.info("Scheduled full ingestion cycle starting")
        try:
            await self.trigger_full_cycle()
        except Exception as e:
            logger.error("Scheduled ingestion cycle failed: {}", str(e))

    async def _check_orbital_schedule(self) -> None:
        """Scheduled job: check Sentinel-1 orbital passes.

        In production, this would query the Copernicus mission planning
        API to check for upcoming Sentinel-1 passes over Indian EEZ.
        For now, it maintains a simplified schedule based on known
        repeat cycles (12-day for S1A, 12-day for S1B offset by 6 days).
        """
        now = datetime.now(UTC)

        # Simplified: S1A revisit every ~6 days over Indian EEZ
        # S1B fills the gap, giving ~3-day effective revisit
        day_of_year = now.timetuple().tm_yday

        # Trigger SAR-specific processing every ~3 days
        if day_of_year % 3 == 0:
            logger.info("Orbital schedule: Sentinel-1 pass expected today")
            self._orbital_schedule.append(
                {
                    "time": now.isoformat(),
                    "satellite": "S1A" if day_of_year % 6 == 0 else "S1B",
                    "status": "expected",
                }
            )

            # Keep only last 30 entries
            self._orbital_schedule = self._orbital_schedule[-30:]
