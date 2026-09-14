"""Tiled SAR windows with a context halo, plus recombination.

AGENTS.md mandates tiled SAR inference. The naive way to tile — cut the raster
into independent chips and label each one — produces two artefacts: a slick
straddling a tile edge becomes two half-detections, and every operator with a
neighbourhood (the Lee-sigma filter, the 151-px background box filter,
morphological open/close) is wrong near a tile border.

This module tiles in a way that makes those two problems disappear by
construction:

* **Non-overlapping cores, overlapping reads.** The scene is partitioned into
  non-overlapping *cores* of ``tile_size - overlap`` pixels. Each core is then
  read back with a halo of ``overlap // 2`` pixels on every side, so the tile
  actually processed is ``tile_size`` wide. Every operator therefore sees real
  neighbours, and no pixel is processed twice.
* **The halo is sized from the operators, not guessed.**
  :func:`context_radius` returns how far the detector reaches (half the largest
  window, plus the morphological iterations). The plan enforces
  ``overlap >= 2 * context + 2``; if the requested tile is too small to hold
  that context the plan collapses to a single tile rather than silently
  degrading.
* **Recombination is a write-back, not a merge.** Each tile writes only its
  core into a full-scene buffer, so the result is bit-identical to an untiled
  run. Connected components are then labelled on the *whole* stitched mask,
  which is what merges a slick that crosses a tile boundary into one polygon.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator

from app.processors.deterministic import DetectionConfig

_SINGLE_TILE_MARGIN = 32  # px; a tile must have this much core left after halos


@dataclass(frozen=True)
class TileWindow:
    """One tile: the region it owns (core) and the region it reads (core+halo)."""

    index: int
    row0: int
    col0: int
    row1: int  # exclusive
    col1: int  # exclusive
    halo: int
    height: int
    width: int

    @property
    def core(self) -> tuple[slice, slice]:
        """Slice of the full-scene buffer this tile writes into."""
        return slice(self.row0, self.row1), slice(self.col0, self.col1)

    @property
    def read(self) -> tuple[slice, slice]:
        """Slice of the full-scene buffer this tile reads (core plus halo)."""
        return (
            slice(max(0, self.row0 - self.halo), min(self.height, self.row1 + self.halo)),
            slice(max(0, self.col0 - self.halo), min(self.width, self.col1 + self.halo)),
        )

    @property
    def core_in_read(self) -> tuple[slice, slice]:
        """Slice *within the read buffer* holding the core (write-back region)."""
        r0 = self.row0 - max(0, self.row0 - self.halo)
        c0 = self.col0 - max(0, self.col0 - self.halo)
        return slice(r0, r0 + (self.row1 - self.row0)), slice(c0, c0 + (self.col1 - self.col0))

    @property
    def shape(self) -> tuple[int, int]:
        r, c = self.read
        return r.stop - r.start, c.stop - c.start

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "core": [self.row0, self.col0, self.row1, self.col1],
            "halo": self.halo,
            "read_shape": list(self.shape),
        }


@dataclass(frozen=True)
class TilePlan:
    """A concrete tiling of one scene."""

    height: int
    width: int
    tile_size: int
    overlap: int  # effective (may be raised to cover the required context)
    context: int  # operator reach in px
    stride: int
    grid_rows: int
    grid_cols: int
    single_tile: bool
    note: str | None = None

    @property
    def n_tiles(self) -> int:
        return self.grid_rows * self.grid_cols

    def windows(self) -> Iterator[TileWindow]:
        halo = self.overlap // 2
        idx = 0
        for row0 in range(0, self.height, self.stride):
            row1 = min(row0 + self.stride, self.height)
            for col0 in range(0, self.width, self.stride):
                col1 = min(col0 + self.stride, self.width)
                yield TileWindow(
                    index=idx,
                    row0=row0,
                    col0=col0,
                    row1=row1,
                    col1=col1,
                    halo=halo,
                    height=self.height,
                    width=self.width,
                )
                idx += 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.n_tiles > 1,
            "tile_size": self.tile_size,
            "overlap": self.overlap,
            "context_radius_px": self.context,
            "stride": self.stride,
            "grid": [self.grid_rows, self.grid_cols],
            "n_tiles": self.n_tiles,
            "note": self.note,
        }


def context_radius(config: DetectionConfig) -> int:
    """How many pixels of context the deterministic detector reaches.

    ``big_window`` (the adaptive-threshold background box) dominates: its box
    filter needs ``big_window // 2`` pixels on each side. The Lee-sigma filter
    and the morphological clean-up are far shallower but are included so the
    halo stays correct if the config is retuned.
    """
    return int(
        max(
            config.big_window // 2 + 1,
            config.local_window // 2 + 1,
            config.lee_window // 2 + 1,
            config.open_iter + config.close_iter + 1,
        )
    )


def plan_tiles(
    height: int,
    width: int,
    *,
    tile_size: int = 1024,
    overlap: int = 192,
    config: DetectionConfig | None = None,
) -> TilePlan:
    """Build a tile plan, widening ``overlap`` until it covers the operator context.

    If ``tile_size`` cannot hold the required context plus a usable core, the
    plan collapses to a single tile covering the whole scene and says so in
    ``note`` — a correct untiled run beats a tiled run with border artefacts.
    """
    cfg = config or DetectionConfig()
    context = context_radius(cfg)
    needed = 2 * context + 2

    if tile_size < needed + _SINGLE_TILE_MARGIN:
        return TilePlan(
            height=height,
            width=width,
            tile_size=max(height, width),
            overlap=0,
            context=context,
            stride=max(height, width),
            grid_rows=1,
            grid_cols=1,
            single_tile=True,
            note=(
                f"tile_size={tile_size} cannot hold the {context} px operator context "
                f"(needs >={needed + _SINGLE_TILE_MARGIN}); running untiled instead of "
                "producing border artefacts"
            ),
        )

    effective_overlap = max(int(overlap), needed)
    if effective_overlap != overlap and effective_overlap > 0:
        note = (
            f"overlap raised {overlap} -> {effective_overlap} px to cover the "
            f"{context} px operator context"
        )
    else:
        note = None
    # Never let the halo eat the whole tile.
    effective_overlap = min(effective_overlap, tile_size - _SINGLE_TILE_MARGIN)

    stride = tile_size - effective_overlap
    grid_rows = -(-height // stride)
    grid_cols = -(-width // stride)
    return TilePlan(
        height=height,
        width=width,
        tile_size=tile_size,
        overlap=effective_overlap,
        context=context,
        stride=stride,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        single_tile=False,
        note=note,
    )


@dataclass
class TileStats:
    """Bookkeeping for a tiled pass, so callers can log what actually happened."""

    tiles: int = 0
    tile_shapes: list[list[int]] = field(default_factory=list)

    def record(self, window: TileWindow) -> None:
        self.tiles += 1
        self.tile_shapes.append(list(window.shape))
