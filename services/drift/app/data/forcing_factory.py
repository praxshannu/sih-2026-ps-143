"""Forcing provider factory with explicit, per-field provenance.

Policy (per super-prompt + AGENTS.md No-Fabrication Policy)
-----------------------------------------------------------
* Real sources are always preferred, and they are chosen **per field** —
  currents and wind are independent decisions. A missing ERA5 key must never
  also throw away perfectly good CMEMS currents (the previous version did
  exactly that: it gated both fields on one credential check).
* Every selection is recorded: source, provider class, real-vs-synthetic,
  status, and the coverage window actually available.
* Synthetic data is opt-in. `allow_synthetic=False` raises a typed
  `MissingWindForcingError` / `MissingCurrentForcingError` instead of quietly
  serving mock fields. `allow_synthetic=True` (the default, for backwards
  compatibility) serves them but the returned record labels them and applies
  a confidence penalty — see `forcing_selection`.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from loguru import logger

from ..errors import MissingCurrentForcingError, MissingWindForcingError
from .forcing_selection import (
    REAL_CONFIGURED,
    REAL_STAGED,
    SYNTHETIC,
    FieldSelection,
    ForcingSelection,
    coverage_from_dataset,
)

if TYPE_CHECKING:
    # Type-checking sees exactly one binding per name. The runtime fallback
    # below binds the same classes a second time (via the `app.data.*` module
    # name the pytest path bootstrap creates), which mypy reads as a
    # redefinition; the `else` branch keeps that behaviour at runtime.
    from .cmems_provider import CMEMSCurrentProvider
    from .composite_provider import CompositeForcingProvider
    from .era5_provider import ERA5WindProvider
    from .forcing_base import EnvironmentalForcingProvider
    from .gfs_provider import NOAAGFSWindProvider
    from .mock_forcing import MockEnvironmentalForcingProvider
else:
    try:  # normal package import (container / app suite)
        from .cmems_provider import CMEMSCurrentProvider
        from .composite_provider import CompositeForcingProvider
        from .era5_provider import ERA5WindProvider
        from .forcing_base import EnvironmentalForcingProvider
        from .gfs_provider import NOAAGFSWindProvider
        from .mock_forcing import MockEnvironmentalForcingProvider
    except ImportError:  # direct file loading (pytest path bootstrap)
        from app.data.cmems_provider import CMEMSCurrentProvider
        from app.data.composite_provider import CompositeForcingProvider
        from app.data.era5_provider import ERA5WindProvider
        from app.data.forcing_base import EnvironmentalForcingProvider
        from app.data.gfs_provider import NOAAGFSWindProvider
        from app.data.mock_forcing import MockEnvironmentalForcingProvider


def _have_copernicus() -> bool:
    u = (
        os.getenv("COPERNICUSMARINE_SERVICE_USERNAME")
        or os.getenv("COPERNICUS_MARINE_USERNAME")
        or os.getenv("COPERNICUS_USER")
    )
    p = (
        os.getenv("COPERNICUSMARINE_SERVICE_PASSWORD")
        or os.getenv("COPERNICUS_MARINE_PASSWORD")
        or os.getenv("COPERNICUS_PASSWORD")
    )
    return bool(u and p)


def _exists(path: Optional[str]) -> bool:
    return bool(path) and os.path.exists(str(path))


def _select_current(
    cmems_path: Optional[str],
    allow_synthetic: bool,
) -> Tuple[EnvironmentalForcingProvider, FieldSelection]:
    """Pick the best real current source; only then consider synthetic."""
    if _exists(cmems_path):
        provider = CMEMSCurrentProvider(dataset_path=str(cmems_path))
        start, end = provider.coverage_window()
        logger.info("Forcing/current: local CMEMS file {}", cmems_path)
        return provider, FieldSelection(
            field="current",
            source="cmems_local_file",
            provider=type(provider).__name__,
            is_real=True,
            status=REAL_STAGED,
            reason=f"Opened local CMEMS NetCDF {cmems_path}",
            dataset_id=provider.product_id,
            coverage_start_utc=start,
            coverage_end_utc=end,
        )

    probe = CMEMSCurrentProvider()
    if probe.live_available():
        logger.info("Forcing/current: live CMEMS (credentials configured)")
        return probe, FieldSelection(
            field="current",
            source="cmems",
            provider=type(probe).__name__,
            is_real=True,
            status=REAL_CONFIGURED,
            reason="Copernicus Marine credentials configured; currents staged on demand",
            dataset_id=probe.product_id,
        )

    if not allow_synthetic:
        raise MissingCurrentForcingError(
            "No real ocean-current forcing available: no local CMEMS file and no "
            "Copernicus Marine credentials (COPERNICUSMARINE_SERVICE_USERNAME/"
            "PASSWORD). Synthetic currents refused — set allow_synthetic=True to "
            "accept a labelled, confidence-penalised mock field.",
            reason="cmems_unavailable_synthetic_refused",
        )

    logger.warning("Forcing/current: NO real source — labelled synthetic_mock")
    return MockEnvironmentalForcingProvider(), FieldSelection(
        field="current",
        source="synthetic_mock",
        provider="MockEnvironmentalForcingProvider",
        is_real=False,
        status=SYNTHETIC,
        reason="No local CMEMS file and no Copernicus Marine credentials; "
        "synthetic currents substituted (never evidence).",
    )


def _select_wind(
    era5_path: Optional[str],
    gfs_path: Optional[str],
    window: Optional[Tuple[datetime, datetime]],
    allow_synthetic: bool,
) -> Tuple[EnvironmentalForcingProvider, FieldSelection]:
    """Pick the best real wind source; only then consider synthetic."""
    if _exists(era5_path):
        provider: EnvironmentalForcingProvider = ERA5WindProvider(dataset_path=str(era5_path))
        start, end = provider.coverage_window()
        logger.info("Forcing/wind: local ERA5 file {}", era5_path)
        return provider, FieldSelection(
            field="wind",
            source="era5_local_file",
            provider=type(provider).__name__,
            is_real=True,
            status=REAL_STAGED,
            reason=f"Opened local ERA5 NetCDF {era5_path}",
            dataset_id="reanalysis-era5-single-levels",
            coverage_start_utc=start,
            coverage_end_utc=end,
        )

    era5_probe = ERA5WindProvider()
    if era5_probe.live_available():
        logger.info("Forcing/wind: live ERA5 (CDS credentials configured)")
        return era5_probe, FieldSelection(
            field="wind",
            source="era5",
            provider=type(era5_probe).__name__,
            is_real=True,
            status=REAL_CONFIGURED,
            reason="CDS (CDSAPI_URL + CDSAPI_KEY) configured; ERA5 10 m wind fetched on demand",
            dataset_id="reanalysis-era5-single-levels",
        )

    if _exists(gfs_path):
        provider = NOAAGFSWindProvider(dataset_path=str(gfs_path))
        start, end = provider.coverage_window()
        logger.info("Forcing/wind: local GFS file {}", gfs_path)
        return provider, FieldSelection(
            field="wind",
            source="gfs_local_file",
            provider=type(provider).__name__,
            is_real=True,
            status=REAL_STAGED,
            reason=f"Opened local GFS NetCDF {gfs_path}",
            dataset_id="NOAA_GFS_0p25",
            coverage_start_utc=start,
            coverage_end_utc=end,
        )

    gfs_reason = "no window supplied, so NOMADS retention could not be checked"
    if window is not None:
        try:
            from ..sources.gfs import supports_window

            ok, why = supports_window(window[0], window[1])
            if ok:
                logger.info("Forcing/wind: live NOAA GFS (window within NOMADS retention)")
                return NOAAGFSWindProvider(), FieldSelection(
                    field="wind",
                    source="gfs",
                    provider="NOAAGFSWindProvider",
                    is_real=True,
                    status=REAL_CONFIGURED,
                    reason=f"NOAA GFS via NOMADS filter ({why})",
                    dataset_id="NOAA_GFS_0p25",
                )
            gfs_reason = f"GFS refused the window: {why}"
        except Exception as exc:  # noqa: BLE001 - GFS is a fallback, not a hard dep
            gfs_reason = f"GFS availability check failed: {str(exc)[:160]}"

    if not allow_synthetic:
        raise MissingWindForcingError(
            "No real wind forcing available: no local ERA5/GFS file, no CDS "
            f"credentials, and {gfs_reason}. Synthetic wind refused — set "
            "allow_synthetic=True to accept a labelled, confidence-penalised "
            "mock field.",
            reason="wind_unavailable_synthetic_refused",
        )

    logger.warning("Forcing/wind: NO real source ({}) — labelled synthetic_mock", gfs_reason)
    return MockEnvironmentalForcingProvider(), FieldSelection(
        field="wind",
        source="synthetic_mock",
        provider="MockEnvironmentalForcingProvider",
        is_real=False,
        status=SYNTHETIC,
        reason=f"No local ERA5/GFS file, no CDS credentials, and {gfs_reason}; "
        "synthetic wind substituted (never evidence).",
    )


def _legacy_label(wind: FieldSelection, current: FieldSelection) -> str:
    """Top-level `forcing_source` label.

    Values are part of the existing UI contract (`live_cmems_era5`,
    `composite_gfs_mock`, `local_files`, `synthetic_mock`), so they are kept
    stable even though the per-field detail is now richer.
    """
    if wind.is_real and current.is_real:
        if wind.status == REAL_STAGED and current.status == REAL_STAGED:
            return "local_files"
        return f"live_{current.source}_{wind.source}"
    if not wind.is_real and not current.is_real:
        # Legacy name retained: the free GFS wind source was preferred but
        # unusable, so a labelled mock stands in for the whole run.
        return "composite_gfs_mock"
    return f"composite_{wind.source}_{current.source}"


def build_forcing_selection(
    cmems_path: str | None = None,
    era5_path: str | None = None,
    gfs_path: str | None = None,
    *,
    window: Tuple[datetime, datetime] | None = None,
    allow_synthetic: bool = True,
) -> ForcingSelection:
    """Resolve both forcing fields and return the recorded selection."""
    current_provider, current_sel = _select_current(cmems_path, allow_synthetic)
    wind_provider, wind_sel = _select_wind(era5_path, gfs_path, window, allow_synthetic)
    selection = ForcingSelection(
        wind=wind_sel,
        current=current_sel,
        forcing_source=_legacy_label(wind_sel, current_sel),
    )
    if selection.any_synthetic:
        logger.warning(
            "Forcing selection is NOT fully real ({}): confidence will be penalised",
            ", ".join(selection.synthetic_fields),
        )
    return selection


def build_forcing_provider(
    cmems_path: str | None = None,
    era5_path: str | None = None,
    gfs_path: str | None = None,
    *,
    window: Tuple[datetime, datetime] | None = None,
    allow_synthetic: bool = True,
) -> Tuple[EnvironmentalForcingProvider, Dict[str, Any]]:
    """Build the best available forcing provider.

    Returns (provider, provenance) where provenance always contains
    `forcing_source` in {"live_cmems_era5", "composite_gfs_mock",
    "synthetic_mock", "local_files", ...} plus per-source origins, per-field
    real/synthetic flags, coverage windows and confidence penalty. Callers
    must echo this in API responses.

    Raises
    ------
    MissingCurrentForcingError / MissingWindForcingError
        When `allow_synthetic=False` and no real source exists for that
        field. Nothing is fabricated in that case.
    """
    current_provider, current_sel = _select_current(cmems_path, allow_synthetic)
    wind_provider, wind_sel = _select_wind(era5_path, gfs_path, window, allow_synthetic)

    # Concrete type, not the ABC: `selection` is CompositeForcingProvider's own
    # attribute (it is what CompositeForcingProvider.metadata() reports), so
    # annotating the base class here would hide the write from mypy.
    provider = CompositeForcingProvider(current_provider, wind_provider)
    selection = ForcingSelection(
        wind=wind_sel,
        current=current_sel,
        forcing_source=_legacy_label(wind_sel, current_sel),
    )
    provider.selection = selection
    logger.info(
        "Forcing: source={} wind={}(real={}) current={}(real={})",
        selection.forcing_source,
        wind_sel.source,
        wind_sel.is_real,
        current_sel.source,
        current_sel.is_real,
    )
    return provider, selection.to_dict()


__all__ = [
    "build_forcing_provider",
    "build_forcing_selection",
    "coverage_from_dataset",
    "ForcingSelection",
    "FieldSelection",
]
