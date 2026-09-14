"""
Composite Environmental Forcing Provider for SENTINEL.
Adapted from OceanTrace agent2/adapters/composite_forcing.py.

Pairs independent ocean current and atmospheric wind providers into a unified forcing engine.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from .forcing_base import EnvironmentalForcingProvider
from .forcing_selection import ForcingSelection


class CompositeForcingProvider(EnvironmentalForcingProvider):
    """
    Blends an ocean current provider with an atmospheric wind provider.
    """

    def __init__(
        self,
        current_provider: EnvironmentalForcingProvider,
        wind_provider: EnvironmentalForcingProvider,
    ):
        self.current_provider = current_provider
        self.wind_provider = wind_provider
        # Populated by the factory: which source won for each field, and
        # whether that source is real. Never None once built by the factory.
        self.selection: ForcingSelection | None = None

    def get_current_vectors(
        self, lons: np.ndarray, lats: np.ndarray, timestamp: datetime
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self.current_provider.get_current_vectors(lons, lats, timestamp)

    def get_wind_vectors(
        self, lons: np.ndarray, lats: np.ndarray, timestamp: datetime
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self.wind_provider.get_wind_vectors(lons, lats, timestamp)

    def get_sst(self, lon: float, lat: float, timestamp: datetime) -> float:
        try:
            return float(self.current_provider.get_sst(lon, lat, timestamp))
        except Exception:
            return 26.0

    def to_opendrift_readers(self) -> Optional[List[Any]]:
        """Aggregates OpenDrift readers from both sub-providers (current + wind)."""
        readers: List[Any] = []
        for provider in (self.current_provider, self.wind_provider):
            sub = provider.to_opendrift_readers()
            if sub:
                readers.extend(sub)
        return readers or None

    def prepare_for_region(self, spill: Any) -> None:
        """Stages remote forcing for both sub-providers (current + wind)."""
        for provider in (self.current_provider, self.wind_provider):
            provider.prepare_for_region(spill)

    def coverage_window(self) -> tuple[str | None, str | None]:
        """Intersection of the two sub-providers' coverage.

        Wind and currents rarely share a grid; the honest answer for a
        composite is the overlap actually usable by a simulation.
        """
        ws, we = self.wind_provider.coverage_window()
        cs, ce = self.current_provider.coverage_window()
        if ws is None or cs is None:
            return None, None
        return max(ws, cs), min(we or ws, ce or cs)

    def metadata(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "provider": "CompositeForcingProvider",
            "current_source": self.current_provider.metadata(),
            "wind_source": self.wind_provider.metadata(),
        }
        if self.selection is not None:
            out["selection"] = self.selection.to_dict()
        return out
