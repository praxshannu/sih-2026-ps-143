"""Look-alike discriminator for false positive rejection.

Rejects low-confidence detections that are likely wind-induced features
or natural ocean surface patterns rather than genuine oil spills.
Uses wind speed, GLCM texture features, and spatial context heuristics.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from loguru import logger
from scipy import ndimage

from app.schemas import LookalikeAnalysis, WindData


class LookalikeFilter:
    """Discriminate genuine oil spills from look-alike features.

    Rejection criteria:
      1. Low-wind regime (<3 m/s) produces dark spots from capillary waves
         that mimic oil dampening — reject unless texture confirms.
      2. High texture homogeneity within the detected region suggests
         natural slick (biogenic film) rather than mineral oil.
      3. Spatial context: proximity to known shorelines or shipping lanes
         increases spill likelihood.

    Args:
        wind_threshold_ms: minimum wind speed to consider detection reliable.
        texture_homogeneity_threshold: GLCM homogeneity threshold for
            flagging natural films.
        min_texture_contrast: minimum GLCM contrast for real oil.
        context_buffer_m: buffer around coastline for context scoring.
    """

    def __init__(
        self,
        wind_threshold_ms: float = 3.0,
        texture_homogeneity_threshold: float = 0.7,
        min_texture_contrast: float = 0.1,
        context_buffer_m: float = 5000.0,
    ) -> None:
        self.wind_threshold_ms = wind_threshold_ms
        self.texture_homogeneity_threshold = texture_homogeneity_threshold
        self.min_texture_contrast = min_texture_contrast
        self.context_buffer_m = context_buffer_m

    def compute_texture_features(
        self, sar_patch: np.ndarray
    ) -> dict[str, float]:
        """Compute GLCM texture features for a SAR patch.

        Args:
            sar_patch: (H, W) float32 SAR intensity patch.

        Returns:
            Dict with keys: homogeneity, contrast, energy, correlation.
        """
        # Quantise to 8-bit for GLCM computation
        if sar_patch.max() - sar_patch.min() < 1e-8:
            return {"homogeneity": 1.0, "contrast": 0.0, "energy": 1.0, "correlation": 0.0}

        normed = ((sar_patch - sar_patch.min()) / (sar_patch.max() - sar_patch.min()) * 255).astype(np.uint8)

        # Compute GLCM at 0, 45, 90, 135 degrees
        offsets = [(0, 1), (1, 1), (1, 0), (1, -1)]
        glcm_sum = np.zeros((256, 256), dtype=np.float64)

        for dy, dx in offsets:
            rows, cols = normed.shape
            r1, r2 = max(0, -dy), min(rows, rows - dy)
            c1, c2 = max(0, -dx), min(cols, cols - dx)
            if r1 >= r2 or c1 >= c2:
                continue
            for shift_y in range(max(0, dy), min(rows, rows + dy)):
                for shift_x in range(max(0, dx), min(cols, cols + dx)):
                    i_val = normed[shift_y, shift_x]
                    j_y = shift_y - dy
                    j_x = shift_x - dx
                    if 0 <= j_y < rows and 0 <= j_x < cols:
                        j_val = normed[j_y, j_x]
                        glcm_sum[i_val, j_val] += 1

        # Normalise
        total = glcm_sum.sum()
        if total > 0:
            glcm = glcm_sum / total
        else:
            return {"homogeneity": 1.0, "contrast": 0.0, "energy": 1.0, "correlation": 0.0}

        # Haralick features
        i_indices, j_indices = np.indices(glcm.shape)
        homogeneity = float(np.sum(glcm / (1.0 + (i_indices - j_indices) ** 2)))
        contrast = float(np.sum(glcm * (i_indices - j_indices) ** 2))
        energy = float(np.sqrt(np.sum(glcm ** 2)))

        # Correlation
        mu_i = np.sum(i_indices * glcm)
        mu_j = np.sum(j_indices * glcm)
        sigma_i = np.sqrt(np.sum((i_indices - mu_i) ** 2 * glcm))
        sigma_j = np.sqrt(np.sum((j_indices - mu_j) ** 2 * glcm))
        if sigma_i > 0 and sigma_j > 0:
            correlation = float(np.sum((i_indices - mu_i) * (j_indices - mu_j) * glcm) / (sigma_i * sigma_j))
        else:
            correlation = 0.0

        return {
            "homogeneity": homogeneity,
            "contrast": contrast,
            "energy": energy,
            "correlation": correlation,
        }

    def wind_regime_analysis(
        self,
        wind: Optional[WindData],
        sar_mask: np.ndarray,
    ) -> tuple[float, list[str]]:
        """Analyse wind regime for look-alike probability.

        Returns:
            Tuple of (score 0-1 where 1 = likely lookalike, reasons list).
        """
        reasons: list[str] = []
        score = 0.0

        if wind is None:
            reasons.append("No wind data provided, cannot assess wind regime")
            return 0.3, reasons

        wind_speed = wind.speed

        if wind_speed < self.wind_threshold_ms:
            # Low wind: dark spots are more likely capillary wave effects
            score += 0.5
            reasons.append(
                f"Low wind speed ({wind_speed:.1f} m/s < {self.wind_threshold_ms} m/s) "
                "increases look-alike probability"
            )
        elif wind_speed < 6.0:
            # Moderate wind: some uncertainty
            score += 0.15
            reasons.append(
                f"Moderate wind ({wind_speed:.1f} m/s) — marginal detection confidence"
            )
        else:
            # High wind: dark spots are less likely to be natural
            score -= 0.2
            reasons.append(f"Good wind ({wind_speed:.1f} m/s) supports genuine detection")

        return max(0.0, min(1.0, score)), reasons

    def texture_analysis(
        self,
        sar_patch: np.ndarray,
        spill_mask: np.ndarray,
    ) -> tuple[float, list[str]]:
        """Analyse texture within detected spill region.

        Returns:
            Tuple of (score 0-1 where 1 = likely lookalike, reasons list).
        """
        reasons: list[str] = []
        score = 0.0

        # Extract spill region
        spill_pixels = sar_patch[spill_mask > 0]
        if len(spill_pixels) < 50:
            reasons.append("Spill region too small for texture analysis")
            return 0.2, reasons

        features = self.compute_texture_features(sar_patch)

        # High homogeneity suggests biogenic film rather than mineral oil
        if features["homogeneity"] > self.texture_homogeneity_threshold:
            score += 0.35
            reasons.append(
                f"High texture homogeneity ({features['homogeneity']:.3f}) "
                "suggests natural/biogenic film"
            )

        # Low contrast suggests uniform dampening (wind-induced)
        if features["contrast"] < self.min_texture_contrast:
            score += 0.25
            reasons.append(
                f"Low texture contrast ({features['contrast']:.3f}) "
                "indicates possible wind-induced feature"
            )

        # High energy with low correlation suggests noise
        if features["energy"] > 0.8 and abs(features["correlation"]) < 0.1:
            score += 0.15
            reasons.append("High energy with low correlation — possible noise pattern")

        return max(0.0, min(1.0, score)), reasons

    def context_analysis(
        self,
        spill_mask: np.ndarray,
        shoreline_mask: Optional[np.ndarray] = None,
        shipping_lane_mask: Optional[np.ndarray] = None,
    ) -> tuple[float, list[str]]:
        """Analyse spatial context (proximity to shore/shipping lanes).

        Returns:
            Tuple of (score 0-1 where 1 = likely lookalike, reasons list).
        """
        reasons: list[str] = []
        score = 0.0

        if shoreline_mask is not None:
            # Calculate distance to nearest shoreline
            distance_map = ndimage.distance_transform_edt(
                shoreline_mask == 0
            )
            min_dist = distance_map[spill_mask > 0].min()

            if min_dist < 100:  # pixels
                score += 0.2
                reasons.append(f"Very close to shoreline ({min_dist:.0f} px)")
            elif min_dist > 500:
                score -= 0.1
                reasons.append(f"Offshore detection ({min_dist:.0f} px from shore)")

        if shipping_lane_mask is not None:
            overlap = np.logical_and(spill_mask > 0, shipping_lane_mask > 0).sum()
            spill_area = (spill_mask > 0).sum()
            if spill_area > 0 and overlap / spill_area > 0.5:
                score += 0.15
                reasons.append("Overlap with shipping lane — possible bilge discharge")

        return max(0.0, min(1.0, score)), reasons

    def assess(
        self,
        sar_patch: np.ndarray,
        spill_mask: np.ndarray,
        wind: Optional[WindData] = None,
        shoreline_mask: Optional[np.ndarray] = None,
        shipping_lane_mask: Optional[np.ndarray] = None,
    ) -> LookalikeAnalysis:
        """Combined look-alike assessment.

        Args:
            sar_patch: (H, W) SAR intensity patch.
            spill_mask: (H, W) binary mask of detected spill.
            wind: optional ERA5 wind data.
            shoreline_mask: optional binary shoreline mask.
            shipping_lane_mask: optional binary shipping lane mask.

        Returns:
            LookalikeAnalysis with score, flag, and reasons.
        """
        all_reasons: list[str] = []

        wind_score, wind_reasons = self.wind_regime_analysis(wind, spill_mask)
        all_reasons.extend(wind_reasons)

        texture_score, texture_reasons = self.texture_analysis(sar_patch, spill_mask)
        all_reasons.extend(texture_reasons)

        context_score, context_reasons = self.context_analysis(
            spill_mask, shoreline_mask, shipping_lane_mask
        )
        all_reasons.extend(context_reasons)

        # Weighted combination
        combined_score = (
            0.4 * wind_score
            + 0.35 * texture_score
            + 0.25 * context_score
        )

        is_lookalike = combined_score > 0.5

        logger.debug(
            f"Lookalike score: {combined_score:.3f} "
            f"(wind={wind_score:.3f}, tex={texture_score:.3f}, "
            f"ctx={context_score:.3f}) -> "
            f"{'LOOKALIKE' if is_lookalike else 'GENUINE'}"
        )

        return LookalikeAnalysis(
            is_lookalike=is_lookalike,
            score=round(combined_score, 4),
            reasons=all_reasons,
        )
