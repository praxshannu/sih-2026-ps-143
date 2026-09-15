"""Raster I/O with the GDAL flags this project actually needs.

The flag that matters
---------------------
``GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR``. By default GDAL scans the
containing directory every time it opens a file, looking for sidecars
(``.aux.xml``, ``.ovr``, ``.msk``). On the external archive — a 2400-entry
exFAT directory on USB — that scan cost **268 ms per open**, and opening is
what dominates a header probe.

With the scan disabled the same open costs **1.8 ms**. That is a 150x
difference, and it is the difference between "validating all 1200 scenes is a
five-minute stall" and "validating all 1200 scenes is two seconds". It is
worth stating plainly because the failure mode is invisible: nothing is wrong,
everything is just slow, and the obvious conclusion is to check fewer files.

The trade-off is real but irrelevant here: we never ship ``.aux.xml`` or
``.ovr`` sidecars, and the archive is a flat set of self-contained GeoTIFFs.
If a future source *does* rely on sidecars, pass ``GDAL_DISABLE_READDIR_ON_OPEN``
explicitly through :func:`raster_env` rather than editing the default.

Everything here silences the Python warnings and the ``rasterio`` logger,
because GDAL emits one ``CPL_AppDefined`` message per 2-band float32 GeoTIFF
about how it will interpret the extra sample. The message is correct and, at
1200 files, useless.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sentinel_core.errors import DataSourceMismatchError
from sentinel_core.logging import quiet_logger

__all__ = [
    "GDAL_ENV",
    "RasterHeader",
    "header_info",
    "open_raster",
    "raster_env",
    "read_bands",
]

#: Applied to every raster operation in SENTINEL unless overridden.
GDAL_ENV: dict[str, str] = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    # Decompressing a tiled LZW GeoTIFF is CPU-bound; let GDAL use the cores.
    "GDAL_NUM_THREADS": "ALL_CPUS",
}


@dataclass(frozen=True)
class RasterHeader:
    """What a raster says about itself, read without touching a pixel."""

    path: Path
    bands: int
    width: int
    height: int
    dtype: str
    crs: str | None
    centre: tuple[float, float] | None
    #: ``(left, bottom, right, top)`` in the raster's own CRS, or ``None`` when
    #: it is not georeferenced. The centroid is not enough to tell whether two
    #: tiles share ground: two 20 km tiles can have nearby centres and still be
    #: disjoint, or distant centres and overlap.
    bounds: tuple[float, float, float, float] | None = None

    @property
    def geo_key(self) -> str:
        """``"lon,lat"`` at 3 dp, or ``""`` when the raster is not georeferenced."""
        if self.centre is None:
            return ""
        return f"{self.centre[0]:.3f},{self.centre[1]:.3f}"


@contextmanager
def raster_env(**overrides: Any) -> Iterator[None]:
    """A ``rasterio.Env`` carrying :data:`GDAL_ENV` plus any overrides."""
    import rasterio

    with rasterio.Env(**{**GDAL_ENV, **overrides}):
        yield


@contextmanager
def open_raster(path: str | Path, **env: Any) -> Iterator[Any]:
    """Open a raster quietly: no GDAL chatter, no Python warnings."""
    import rasterio

    with quiet_logger("rasterio"), warnings.catch_warnings(), raster_env(**env):
        warnings.simplefilter("ignore")
        with rasterio.open(path) as src:
            yield src


def header_info(path: str | Path) -> RasterHeader:
    """Band count, size and geolocation, from the header only.

    Raises:
        DataSourceMismatchError: when the file cannot be opened as a raster.
            A file that is named ``.tif`` and is not one is a data problem, and
            it should name the file rather than surface as a shape error later.
    """
    target = Path(path)
    try:
        with open_raster(target) as src:
            centre: tuple[float, float] | None = None
            extent: tuple[float, float, float, float] | None = None
            if src.crs is not None and src.bounds is not None:
                bounds = src.bounds
                centre = (
                    (bounds.left + bounds.right) / 2.0,
                    (bounds.bottom + bounds.top) / 2.0,
                )
                extent = (bounds.left, bounds.bottom, bounds.right, bounds.top)
            return RasterHeader(
                path=target,
                bands=int(src.count),
                width=int(src.width),
                height=int(src.height),
                dtype=str(src.dtypes[0]) if src.count else "unknown",
                crs=src.crs.to_string() if src.crs is not None else None,
                centre=centre,
                bounds=extent,
            )
    except Exception as exc:
        raise DataSourceMismatchError(
            f"{target.name} could not be opened as a raster: {exc}",
            reason="unreadable_raster",
            context={"path": str(target)},
        ) from exc


def read_bands(path: str | Path) -> Any:
    """Read a raster as a ``(C, H, W)`` float32 array."""
    import numpy as np

    with open_raster(path) as src:
        return src.read().astype(np.float32, copy=False)
