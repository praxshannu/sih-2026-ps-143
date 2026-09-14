"""Tiled inference with overlap stitching for large SAR scenes.

Splits large rasters into overlapping tiles, runs the UNet++ model on each
tile, then stitches results using mean blending at tile boundaries to
eliminate seam artifacts.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger


class TiledInference:
    """Run tiled semantic segmentation with overlap blending.

    Args:
        model: loaded UNet++ model in eval mode.
        tile_size: spatial size of each tile (H, W).
        overlap: pixel overlap between adjacent tiles.
        device: torch device for inference.
        use_amp: whether to use automatic mixed precision.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tile_size: int = 512,
        overlap: int = 64,
        device: torch.device | str = "cpu",
        use_amp: bool = True,
    ) -> None:
        self.model = model
        self.tile_size = tile_size
        self.overlap = overlap
        self.device = torch.device(device)
        self.use_amp = use_amp and self.device.type == "cuda"
        self.stride = tile_size - overlap
        self.expected_channels = self._infer_in_channels(model)

    @staticmethod
    def _infer_in_channels(model: torch.nn.Module) -> int | None:
        """Best-effort read of the model's expected input channels."""
        try:
            for module in model.modules():
                if isinstance(module, torch.nn.Conv2d):
                    return int(module.in_channels)
        except Exception:
            pass
        return None

    def _adapt_channels(self, batch: torch.Tensor) -> torch.Tensor:
        """Slice/pad channel dim to the model's expectation (never crash)."""
        if self.expected_channels is None or batch.shape[1] == self.expected_channels:
            return batch
        have, want = batch.shape[1], self.expected_channels
        if have > want:
            logger.warning(
                "Slicing input channels {} -> {} to match model", have, want
            )
            return batch[:, :want]
        pad = torch.zeros(
            (batch.shape[0], want - have, *batch.shape[2:]),
            dtype=batch.dtype,
            device=batch.device,
        )
        logger.warning("Padding input channels {} -> {} to match model", have, want)
        return torch.cat([batch, pad], dim=1)

    def _compute_pad(self, size: int) -> int:
        """Compute padding to make size divisible by tile_size."""
        remainder = size % self.tile_size
        if remainder == 0:
            return 0
        return self.tile_size - remainder

    def _extract_tiles(
        self, image: np.ndarray
    ) -> tuple[list[tuple[int, int, np.ndarray]], tuple[int, int]]:
        """Extract overlapping tiles from image.

        Returns:
            tiles: list of (row_idx, col_idx, tile_array).
            padded_shape: (H, W) after padding.
        """
        _, h, w = image.shape
        pad_h = self._compute_pad(h)
        pad_w = self._compute_pad(w)

        if pad_h > 0 or pad_w > 0:
            image = np.pad(
                image,
                ((0, 0), (0, pad_h), (0, pad_w)),
                mode="reflect",
            )

        _, ph, pw = image.shape
        tiles = []
        for y in range(0, ph - self.overlap, self.stride):
            y_end = min(y + self.tile_size, ph)
            y_start = max(0, y_end - self.tile_size)
            for x in range(0, pw - self.overlap, self.stride):
                x_end = min(x + self.tile_size, pw)
                x_start = max(0, x_end - self.tile_size)
                tile = image[:, y_start:y_end, x_start:x_end]
                tiles.append((y_start, x_start, tile))

        logger.debug(f"Extracted {len(tiles)} tiles from {ph}x{pw} image")
        return tiles, (ph, pw)

    def _blend_tiles(
        self,
        predictions: list[tuple[int, int, np.ndarray]],
        output_shape: tuple[int, int],
    ) -> np.ndarray:
        """Stitch tile predictions with mean blending.

        Accumulates overlapping predictions and divides by the count
        to get a smooth blended result without seam artifacts.
        """
        h, w = output_shape
        accumulator = np.zeros((1, h, w), dtype=np.float64)
        count = np.zeros((1, h, w), dtype=np.float64)

        for y, x, tile_pred in predictions:
            tile_pred = np.asarray(tile_pred)
            while tile_pred.ndim > 3:
                tile_pred = tile_pred[0]
            th, tw = tile_pred.shape[-2:]
            accumulator[:, y : y + th, x : x + tw] += tile_pred
            count[:, y : y + th, x : x + tw] += 1.0

        count = np.maximum(count, 1.0)
        blended = (accumulator / count).astype(np.float32)
        return blended

    @torch.no_grad()
    def predict(
        self,
        image: np.ndarray,
        batch_size: int = 4,
        num_workers: int = 0,
    ) -> np.ndarray:
        """Run tiled inference on a large image.

        Args:
            image: input array of shape (C, H, W) or (H, W) for single-band.
            batch_size: number of tiles per forward pass.
            num_workers: unused, kept for API compatibility.

        Returns:
            prediction of shape (1, H, W) with raw logits.
        """
        if image.ndim == 2:
            image = image[np.newaxis, ...]
        # Input is channels-first (C, H, W) as produced by MultimodalInputProcessor.

        original_h, original_w = image.shape[1], image.shape[2]
        tiles, (padded_h, padded_w) = self._extract_tiles(image)

        self.model.eval()
        all_predictions: list[tuple[int, int, np.ndarray]] = []

        for i in range(0, len(tiles), batch_size):
            batch_tiles = tiles[i : i + batch_size]
            batch_arrays = np.stack([t[2] for t in batch_tiles], axis=0)
            batch_tensor = torch.from_numpy(batch_arrays).float().to(self.device)
            batch_tensor = self._adapt_channels(batch_tensor)

            if self.use_amp:
                with torch.amp.autocast(device_type="cuda"):
                    logits = self.model(batch_tensor)
            else:
                logits = self.model(batch_tensor)

            preds = logits.cpu().numpy()

            for j, (y, x, _) in enumerate(batch_tiles):
                pred = preds[j : j + 1]  # (1, H, W)
                all_predictions.append((y, x, pred))

        blended = self._blend_tiles(all_predictions, (padded_h, padded_w))
        return blended[:, :original_h, :original_w]
