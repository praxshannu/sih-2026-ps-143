"""Oil spill age estimation using Fay spreading law inverse model.

Implements the Fay (1971) gravitational-viscous spreading model to
estimate spill age from the observed area at detection time.
"""

from __future__ import annotations

import math

from loguru import logger

from app.schemas import AgeEstimate, CurrentData, SpillAgeCategory, WindData


class AgeEstimator:
    """Estimate oil spill age using Fay spreading law.

    The Fay (1971) model describes oil slick spreading in two regimes:
      - Gravitational-inertial: A = π * κ_g * (g * Δρ * V * t² / ρ_w)^(1/2)
      - Gravitational-viscous:  A = π * κ_v * (g * Δρ * V² * t^(3/2) / ν_w)^(1/2)

    For operational use, we use the simplified form:
        A(t) = π * K * t²
    where K is a composite spreading coefficient that depends on:
      - Wind speed (enhances spreading via turbulent diffusion)
      - Oil density and viscosity
      - Sea surface temperature
      - Ocean currents

    The inverse model estimates t = sqrt(A / (π * K)).

    Args:
        oil_density_kgm3: oil density (default 870 kg/m³ for crude).
        water_density_kgm3: seawater density (default 1025 kg/m³).
        oil_viscosity_m2s: oil kinematic viscosity (default 1e-6 m²/s).
        base_spreading_coeff: base K value for calm conditions.
        wind_enhancement_factor: how much wind amplifies K.
        max_age_hours: maximum plausible age for classification.
    """

    # Empirical spreading coefficient categories (m²/s²)
    SPREADING_REGIMES = {
        "calm": 0.01,  # K for calm seas (wind < 3 m/s)
        "moderate": 0.05,  # K for moderate wind (3-8 m/s)
        "rough": 0.12,  # K for rough seas (8-12 m/s)
        "high": 0.25,  # K for high wind (>12 m/s)
    }

    AGE_CATEGORIES = {
        SpillAgeCategory.FRESH: (0.0, 2.0),  # 0-2 hours
        SpillAgeCategory.MILD: (2.0, 12.0),  # 2-12 hours
        SpillAgeCategory.WEATHERED: (12.0, 72.0),  # 12-72 hours
        SpillAgeCategory.HEAVY: (72.0, float("inf")),  # >72 hours
    }

    def __init__(
        self,
        oil_density_kgm3: float = 870.0,
        water_density_kgm3: float = 1025.0,
        oil_viscosity_m2s: float = 1e-6,
        base_spreading_coeff: float = 0.05,
        wind_enhancement_factor: float = 0.02,
        max_age_hours: float = 168.0,
    ) -> None:
        self.oil_density = oil_density_kgm3
        self.water_density = water_density_kgm3
        self.oil_viscosity = oil_viscosity_m2s
        self.K_base = base_spreading_coeff
        self.wind_factor = wind_enhancement_factor
        self.max_age_hours = max_age_hours

    def _compute_spreading_coeff(
        self,
        wind: WindData | None = None,
        current: CurrentData | None = None,
    ) -> float:
        """Compute the Fay spreading coefficient K.

        K depends on wind speed (turbulent mixing) and current (advection).
        """
        K = self.K_base

        if wind is not None:
            wind_speed = wind.speed
            # Wind-enhanced turbulent diffusion
            # Linear model with saturation at high wind
            K += self.wind_factor * min(wind_speed, 15.0)
            logger.debug(
                f"Wind spreading contribution: {self.wind_factor * min(wind_speed, 15.0):.4f} "
                f"(wind={wind_speed:.1f} m/s)"
            )

        if current is not None:
            current_speed = (current.u**2 + current.v**2) ** 0.5
            K += 0.005 * min(current_speed, 2.0)

        # Ensure K is within physical bounds
        K = max(K, 0.001)
        K = min(K, 0.5)

        return K

    def estimate_age(
        self,
        area_m2: float,
        wind: WindData | None = None,
        current: CurrentData | None = None,
    ) -> AgeEstimate:
        """Estimate spill age from observed area.

        Args:
            area_m2: detected spill area in square metres.
            wind: ERA5 wind data at detection time.
            current: ocean current data.

        Returns:
            AgeEstimate with hours, category, and Fay constant.
        """
        K = self._compute_spreading_coeff(wind, current)

        # Inverse Fay model: t = sqrt(A / (π * K))
        if area_m2 <= 0 or K <= 0:
            return AgeEstimate(
                estimated_hours=None,
                category=SpillAgeCategory.FRESH,
                area_at_detection_m2=area_m2,
                fay_constant_k=K,
            )

        t_squared = area_m2 / (math.pi * K)
        t_hours = math.sqrt(t_squared) / 3600.0  # Convert seconds to hours

        # Clamp to max age
        t_hours = min(t_hours, self.max_age_hours)

        # Classify age category
        category = self._classify_age(t_hours)

        logger.debug(
            f"Age estimation: area={area_m2:.0f} m², K={K:.4f}, "
            f"t={t_hours:.1f} h, category={category.value}"
        )

        return AgeEstimate(
            estimated_hours=round(t_hours, 2),
            category=category,
            area_at_detection_m2=round(area_m2, 2),
            fay_constant_k=round(K, 6),
        )

    def _classify_age(self, hours: float) -> SpillAgeCategory:
        """Classify spill age into category."""
        for cat, (lo, hi) in self.AGE_CATEGORIES.items():
            if lo <= hours < hi:
                return cat
        return SpillAgeCategory.HEAVY

    def estimate_multi_scale(
        self,
        polygon_areas_m2: list[float],
        wind: WindData | None = None,
        current: CurrentData | None = None,
    ) -> list[AgeEstimate]:
        """Estimate ages for multiple spill fragments.

        Args:
            polygon_areas_m2: list of individual spill polygon areas.
            wind: ERA5 wind data.
            current: ocean current data.

        Returns:
            List of AgeEstimate, one per polygon.
        """
        K = self._compute_spreading_coeff(wind, current)
        estimates = []

        for area in polygon_areas_m2:
            if area <= 0:
                estimates.append(
                    AgeEstimate(
                        estimated_hours=None,
                        category=SpillAgeCategory.FRESH,
                        area_at_detection_m2=area,
                        fay_constant_k=K,
                    )
                )
                continue

            t_squared = area / (math.pi * K)
            t_hours = math.sqrt(t_squared) / 3600.0
            t_hours = min(t_hours, self.max_age_hours)
            category = self._classify_age(t_hours)

            estimates.append(
                AgeEstimate(
                    estimated_hours=round(t_hours, 2),
                    category=category,
                    area_at_detection_m2=round(area, 2),
                    fay_constant_k=round(K, 6),
                )
            )

        return estimates

    @staticmethod
    def predict_spread_area(
        initial_area_m2: float,
        t_hours: float,
        K: float,
    ) -> float:
        """Forward model: predict area at a future time.

        Args:
            initial_area_m2: area at detection (t=0 reference).
            t_hours: hours since detection.
            K: spreading coefficient.

        Returns:
            Predicted area in m² at t_hours.
        """
        t_seconds = t_hours * 3600.0
        return math.pi * K * t_seconds**2
