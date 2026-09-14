"""Multi-modal input processor for SAR VV + ERA5 wind/current fusion.

Reads Sentinel-1 GeoTIFF bands and optionally fuses ERA5 reanalysis
wind (U10, V10) and ocean current data into a multi-channel tensor
for the UNet++ encoder.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import rasterio
from loguru import logger

from app.schemas import CurrentData, WindData


class MultimodalInputProcessor:
    """Prepare multi-channel input from SAR + auxiliary data.

    The processor produces a (C, H, W) numpy array:
      - Channel 0: SAR VV (or requested polarisation) normalised to [0, 1]
      - Channel 1: SAR VH if available (else zeros)
      - Channel 2: wind U10 component (constant fill)
      - Channel 3: wind V10 component (constant fill)
      - Channel 4: current U component (constant fill)
      - Channel 5: current V component (constant fill)

    If only a single SAR band is available and no wind/current data is
    provided, the output is (1, H, W) for backward compatibility.
    """

    TARGET_DTYPE = np.float32

    def __init__(
        self,
        target_size: tuple[int, int] | None = None,
        normalize: bool = True,
    ) -> None:
        self.target_size = target_size
        self.normalize = normalize

    def read_sar_bands(
        self,
        image_path: str,
        band_selection: str = "VV",
    ) -> dict[str, np.ndarray]:
        """Read SAR bands from a GeoTIFF file.

        Args:
            image_path: path to Sentinel-1 GeoTIFF.
            band_selection: polarisation to extract ("VV", "VH", "VVVH").

        Returns:
            dict with keys 'vv', 'vh' (each HxW float32).
        """
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"SAR image not found: {image_path}")

        bands: dict[str, np.ndarray] = {}

        with rasterio.open(image_path) as src:
            logger.info(
                f"Reading SAR image: {src.width}x{src.height}, {src.count} bands, CRS={src.crs}"
            )

            # Sentinel-1 band naming conventions
            band_map = {}
            for idx in range(1, src.count + 1):
                desc = src.descriptions[idx - 1] if src.descriptions else ""
                name = desc.upper() if desc else f"BAND_{idx}"
                if "VV" in name or idx == 1:
                    band_map["vv"] = idx
                elif "VH" in name or idx == 2:
                    band_map["vh"] = idx

            target_bands = set()
            if "VV" in band_selection.upper() or "VVVH" in band_selection.upper():
                target_bands.add("vv")
            if "VH" in band_selection.upper() or "VVVH" in band_selection.upper():
                target_bands.add("vh")

            for band_key in target_bands:
                band_idx = band_map.get(band_key)
                if band_idx is None and band_key in band_map:
                    band_idx = band_map[band_key]
                elif band_idx is None:
                    band_idx = 1 if band_key == "vv" else min(2, src.count)
                data = src.read(band_idx).astype(self.TARGET_DTYPE)
                bands[band_key] = data
                logger.debug(
                    f"Read {band_key}: shape={data.shape}, "
                    f"range=[{data.min():.4f}, {data.max():.4f}]"
                )

            if not bands:
                data = src.read(1).astype(self.TARGET_DTYPE)
                bands["vv"] = data

        return bands

    def normalize_sar(self, data: np.ndarray) -> np.ndarray:
        """Normalise SAR intensity to [0, 1] using dB scaling.

        Applies: 10 * log10(data) mapped to [0, 1] via min-max.
        """
        if not self.normalize:
            return data.astype(self.TARGET_DTYPE)

        eps = 1e-10
        data_db = 10.0 * np.log10(np.maximum(data, eps))
        p2 = np.percentile(data_db, 2)
        p98 = np.percentile(data_db, 98)
        if p98 - p2 < 1e-6:
            return np.zeros_like(data, dtype=self.TARGET_DTYPE)

        normalised = np.clip((data_db - p2) / (p98 - p2), 0, 1)
        return normalised.astype(self.TARGET_DTYPE)

    def fuse_channels(
        self,
        bands: dict[str, np.ndarray],
        wind: WindData | None = None,
        current: CurrentData | None = None,
    ) -> np.ndarray:
        """Fuse SAR bands with wind/current data into multi-channel array.

        Returns:
            (C, H, W) float32 array.
        """
        h, w = next(iter(bands.values())).shape

        # SAR channels
        vv = self.normalize_sar(bands.get("vv", np.zeros((h, w), dtype=self.TARGET_DTYPE)))
        vh = self.normalize_sar(bands.get("vh", np.zeros((h, w), dtype=self.TARGET_DTYPE)))

        channels = [vv, vh]

        # Wind channels (constant fill)
        if wind is not None:
            u10 = np.full((h, w), wind.u10, dtype=self.TARGET_DTYPE)
            v10 = np.full((h, w), wind.v10, dtype=self.TARGET_DTYPE)
            channels.extend([u10, v10])
        else:
            channels.extend([np.zeros((h, w), dtype=self.TARGET_DTYPE) for _ in range(2)])

        # Current channels (constant fill)
        if current is not None:
            cu = np.full((h, w), current.u, dtype=self.TARGET_DTYPE)
            cv = np.full((h, w), current.v, dtype=self.TARGET_DTYPE)
            channels.extend([cu, cv])
        else:
            channels.extend([np.zeros((h, w), dtype=self.TARGET_DTYPE) for _ in range(2)])

        return np.stack(channels, axis=0)

    def process(
        self,
        image_path: str,
        band_selection: str = "VV",
        wind: WindData | None = None,
        current: CurrentData | None = None,
        target_size: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Full pipeline: read -> fuse -> resize -> return tensor.

        Args:
            image_path: path to Sentinel-1 GeoTIFF.
            band_selection: SAR polarisation to use.
            wind: optional ERA5 wind data.
            current: optional ocean current data.
            target_size: (H, W) to resize output (optional).

        Returns:
            (C, H, W) float32 array ready for model input.
        """
        bands = self.read_sar_bands(image_path, band_selection)
        fused = self.fuse_channels(bands, wind, current)

        size = target_size or self.target_size
        if size is not None and (fused.shape[1] != size[0] or fused.shape[2] != size[1]):
            fused_resized = np.stack(
                [
                    cv2.resize(
                        fused[c],
                        (size[1], size[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )
                    for c in range(fused.shape[0])
                ],
                axis=0,
            )
            logger.debug(f"Resized from {fused.shape} to {fused_resized.shape}")
            return fused_resized

        return fused
