"""Abstract base for SENTINEL environmental forcing providers.

Ported from OceanTrace agent2/adapters/forcing_base.py — interface unchanged,
imports adapted to sentinel layout. All velocities in m/s, eastward (u) /
northward (v).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class EnvironmentalForcingProvider(ABC):
    """Abstract interface for ocean + atmosphere forcing fields."""

    @abstractmethod
    def get_current_vectors(
        self, lons: np.ndarray, lats: np.ndarray, timestamp: datetime
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Ocean surface current (u, v) in m/s at positions + timestamp."""
        raise NotImplementedError

    @abstractmethod
    def get_wind_vectors(
        self, lons: np.ndarray, lats: np.ndarray, timestamp: datetime
    ) -> Tuple[np.ndarray, np.ndarray]:
        """10-m wind (u10, v10) in m/s at positions + timestamp."""
        raise NotImplementedError

    def get_current(
        self, lon: float, lat: float, timestamp: datetime
    ) -> Tuple[float, float]:
        """Single-point convenience wrapper."""
        u_arr, v_arr = self.get_current_vectors(
            np.array([lon], dtype=np.float64),
            np.array([lat], dtype=np.float64),
            timestamp,
        )
        return float(u_arr[0]), float(v_arr[0])

    def get_ocean_current(
        self, lon: float, lat: float, timestamp: datetime
    ) -> Tuple[float, float]:
        return self.get_current(lon, lat, timestamp)

    def get_wind(
        self, lon: float, lat: float, timestamp: datetime
    ) -> Tuple[float, float]:
        u_arr, v_arr = self.get_wind_vectors(
            np.array([lon], dtype=np.float64),
            np.array([lat], dtype=np.float64),
            timestamp,
        )
        return float(u_arr[0]), float(v_arr[0])

    def get_surface_wind(
        self, lon: float, lat: float, timestamp: datetime
    ) -> Tuple[float, float]:
        return self.get_wind(lon, lat, timestamp)

    def get_sst(self, lon: float, lat: float, timestamp: datetime) -> float:
        """Sea-surface temperature in °C (default 26.0)."""
        return 26.0

    def to_opendrift_readers(self) -> Optional[List[Any]]:
        """OpenDrift readers, or None for synthetic/mock providers."""
        return None

    def coverage_window(self) -> Tuple[Optional[str], Optional[str]]:
        """(start_utc, end_utc) actually covered by the staged data.

        (None, None) means "not staged" — the caller must then say so instead
        of implying coverage it does not have.
        """
        return None, None

    def prepare_for_region(self, spill: Any) -> None:
        """Optional hook to stage remote forcing before simulation."""
        return None

    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """Provenance + diagnostics."""
        raise NotImplementedError
