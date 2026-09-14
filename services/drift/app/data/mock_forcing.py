"""Labelled mock environmental forcing for SENTINEL drift.

Never presented as live data. Used only when credentials are absent and
explicitly labelled `synthetic_mock` in API responses.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Tuple

import numpy as np

from .forcing_base import EnvironmentalForcingProvider


class MockEnvironmentalForcingProvider(EnvironmentalForcingProvider):
    """Deterministic synthetic currents + winds for offline/demo runs."""

    def __init__(
        self,
        current_u: float = 0.15,
        current_v: float = 0.05,
        wind_u: float = 4.0,
        wind_v: float = 2.0,
    ) -> None:
        self.current_u = float(current_u)
        self.current_v = float(current_v)
        self.wind_u = float(wind_u)
        self.wind_v = float(wind_v)

    def get_current_vectors(
        self, lons: np.ndarray, lats: np.ndarray, timestamp: datetime
    ) -> Tuple[np.ndarray, np.ndarray]:
        n = np.asarray(lons).shape[0] if np.asarray(lons).ndim else 1
        # Gentle spatial variation so K_ij gradients are non-degenerate.
        lon_a = np.asarray(lons, dtype=np.float64)
        lat_a = np.asarray(lats, dtype=np.float64)
        u = self.current_u + 0.02 * np.sin(np.radians(lon_a * 3.0))
        v = self.current_v + 0.02 * np.cos(np.radians(lat_a * 3.0))
        return np.atleast_1d(u.astype(np.float64)), np.atleast_1d(v.astype(np.float64))

    def get_wind_vectors(
        self, lons: np.ndarray, lats: np.ndarray, timestamp: datetime
    ) -> Tuple[np.ndarray, np.ndarray]:
        lon_a = np.asarray(lons, dtype=np.float64)
        lat_a = np.asarray(lats, dtype=np.float64)
        u = self.wind_u + 0.5 * np.sin(np.radians(lon_a * 2.0))
        v = self.wind_v + 0.5 * np.cos(np.radians(lat_a * 2.0))
        return np.atleast_1d(u.astype(np.float64)), np.atleast_1d(v.astype(np.float64))

    def metadata(self) -> Dict[str, Any]:
        return {
            "provider": "MockEnvironmentalForcingProvider",
            "source": "synthetic_mock",
            "data_origin": "synthetic_mock",
            "is_connected": True,
        }
