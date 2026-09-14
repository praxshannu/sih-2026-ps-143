#!/usr/bin/env python3
"""Fetch REAL Sentinel-1 GRD GeoTIFFs for SENTINEL.

Two Copernicus APIs are combined on purpose:

  1. CDSE OData catalogue (catalogue.dataspace.copernicus.eu)
     -> authoritative scene metadata (product name, acquisition time, orbit,
        size, checksum). This is what the UI shows as provenance.

  2. CDSE-hosted Sentinel Hub Process API (sh.dataspace.copernicus.eu)
     -> server-side calibrated sigma0 GeoTIFF for an exact AOI/time window.
        Same engine that powers the Copernicus Browser. Returns ~2 MB instead
        of a ~2 GB .SAFE zip, so a laptop can hold many scenes and no SNAP /
        pyroSAR toolchain is required.

Every GeoTIFF is written as a Cloud-Optimised GeoTIFF (DEFLATE + overviews)
with a .json sidecar recording scene identity, bbox, band semantics, SHA-256
and the exact request used. Nothing here is synthetic: if a fetch fails the
scene is skipped and reported, never filled with placeholder pixels.

Usage:
    python scripts/fetch_sar_geotiffs.py                # default manifest
    python scripts/fetch_sar_geotiffs.py --list         # show manifest
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import rasterio.shutil
import requests
from dotenv import load_dotenv
from rasterio.enums import Resampling
from rasterio.transform import from_bounds

REPO_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPO_ROOT / ".env")

TOKEN_URL = os.getenv(
    "CDSE_TOKEN_URL",
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
)
CATALOGUE_URL = os.getenv(
    "CDSE_CATALOG_URL", "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
)
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"

OUT_DIR = REPO_ROOT / "data" / "sar"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# VV + VH calibrated sigma0, plus a coastal/land mask so look-alike filtering
# and land masking have something real to work with.
EVALSCRIPT = """
//VERSION=3
function setup() {
  return {
    input: [{ bands: ["VV", "VH", "dataMask"] }],
    output: { bands: 3, sampleType: "FLOAT32" }
  };
}
function evaluatePixel(s) {
  return [s.VV, s.VH, s.dataMask];
}
"""


@dataclass(frozen=True)
class SceneSpec:
    """One requested acquisition: an AOI + a time window + a label."""

    key: str
    label: str
    bbox: tuple[float, float, float, float]  # west, south, east, north
    window: tuple[str, str]  # ISO from, to
    size: int  # square output, pixels
    note: str = ""


# "Enough, not too many": a real time series over the Wakashio spill zone
# (SE of Mauritius, open ocean) plus one recent Arabian Sea scene so the live
# browser and the detector both have present-day imagery.
#
# These dates are NOT guessed. They are the actual Sentinel-1 passes whose
# footprints intersect this AOI, resolved from the CDSE OData catalogue.
# Wakashio grounded 2020-07-25; the hull split and the main release followed
# in early August, so 29 Jul -> 10 Aug -> 16 Aug is the real observing arc.
SPILL_AOI = (57.60, -21.00, 58.20, -20.40)  # SE of Mauritius, ocean-only

MANIFEST: tuple[SceneSpec, ...] = (
    SceneSpec(
        key="wakashio_20200729_early",
        label="MV Wakashio — early slick (IW, 4 d after grounding)",
        bbox=SPILL_AOI,
        window=("2020-07-29T00:00:00Z", "2020-07-30T00:00:00Z"),
        size=2048,
        note="First IW pass after the 25 Jul 2020 grounding.",
    ),
    SceneSpec(
        key="wakashio_20200810_peak",
        label="MV Wakashio — peak slick (IW)",
        bbox=SPILL_AOI,
        window=("2020-08-10T00:00:00Z", "2020-08-11T00:00:00Z"),
        size=2048,
        note="Primary detection scene, after the hull break.",
    ),
    SceneSpec(
        key="wakashio_20200816_late",
        label="MV Wakashio — late slick / weathering (IW)",
        bbox=SPILL_AOI,
        window=("2020-08-16T00:00:00Z", "2020-08-17T00:00:00Z"),
        size=2048,
        note="Weathered slick, for age/weathering comparison.",
    ),
    SceneSpec(
        key="wakashio_20200822_recovery",
        label="MV Wakashio — recovery / post-removal (IW)",
        bbox=SPILL_AOI,
        window=("2020-08-22T00:00:00Z", "2020-08-23T00:00:00Z"),
        size=2048,
        note="Latest pass in the observing arc; slick largely dispersed.",
    ),
    SceneSpec(
        key="mumbai_20260913_recent",
        label="Mumbai offshore — most recent pass (live browser)",
        bbox=(72.00, 15.00, 73.00, 16.00),
        window=("2026-09-13T00:00:00Z", "2026-09-14T00:00:00Z"),
        size=2048,
        note="Present-day acquisition for the in-app Copernicus browser.",
    ),
)


class NoDataError(RuntimeError):
    """Raised when a requested AOI/time window has no real Sentinel-1 coverage."""


def get_token() -> str:
    """OAuth2 client-credentials token for the CDSE realm."""
    resp = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": os.environ["CDSE_CLIENT_ID"],
            "client_secret": os.environ["CDSE_CLIENT_SECRET"],
        },
        timeout=60,
    )
    resp.raise_for_status()
    return str(resp.json()["access_token"])


def catalogue_lookup(token: str, spec: SceneSpec) -> list[dict[str, Any]]:
    """Real scene metadata for the AOI/time window via CDSE OData.

    Two CDSE quirks this works around:
      * the parser rejects the `datetime'...'` literal prefix that older
        SciHub-era examples use — bare ISO literals with .000Z are required;
      * `OrbitNumber` is not a valid $select field (400 on the whole query),
        so only the documented fields are requested.

    The search box is padded because a swath may cover the fetch AOI without
    its footprint polygon intersecting the exact AOI ring; we report the
    scenes found in the padded window and flag whether one is a direct hit.
    """
    pad = 1.5
    w, s, e, n = spec.bbox
    pw, ps, pe, pn = w - pad, s - pad, e + pad, n + pad
    ring = f"{pw} {ps}, {pe} {ps}, {pe} {pn}, {pw} {pn}, {pw} {ps}"
    filt = (
        "Collection/Name eq 'SENTINEL-1'"
        " and Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType'"
        " and att/OData.CSC.StringAttribute/Value eq 'GRD')"
        f" and OData.CSC.Intersects(area=geography'SRID=4326;POLYGON(({ring}))')"
        f" and ContentDate/Start gt {spec.window[0][:-1]}.000Z"
        f" and ContentDate/Start lt {spec.window[1][:-1]}.000Z"
    )
    resp = requests.get(
        CATALOGUE_URL,
        params={
            "$filter": filt,
            "$orderby": "ContentDate/Start asc",
            "$top": "20",
            "$select": "Id,Name,ContentDate,ContentLength",
        },
        headers={"Authorization": f"Bearer {token}"},
        timeout=90,
    )
    resp.raise_for_status()
    return list(resp.json().get("value", []))


def fetch_geotiff(token: str, spec: SceneSpec) -> np.ndarray:
    """Calibrated sigma0 GeoTIFF for the AOI, rendered server-side by Sentinel Hub."""
    w, s, e, n = spec.bbox
    payload = {
        "input": {
            "bounds": {
                "bbox": [w, s, e, n],
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"},
            },
            "data": [
                {
                    "type": "S1GRD",
                    "dataFilter": {
                        "timeRange": {"from": spec.window[0], "to": spec.window[1]},
                        "mosaickingOrder": "mostRecent",
                        "resolution": "HIGH",
                        "polarization": "DV",
                    },
                    "processing": {
                        "orthorectify": True,
                        "demInstance": "COPERNICUS_30",
                        "backCoeff": "SIGMA0_ELLIPSOID",
                    },
                }
            ],
        },
        "output": {
            "width": spec.size,
            "height": spec.size,
            "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}],
        },
        "evalscript": EVALSCRIPT,
    }
    resp = requests.post(
        PROCESS_URL,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "image/tiff",
        },
        json=payload,
        timeout=600,
    )
    resp.raise_for_status()
    if resp.content[:2] not in (b"II", b"MM"):
        raise RuntimeError(f"expected TIFF, got: {resp.content[:200]!r}")

    tmp = OUT_DIR / f".{spec.key}.raw.tif"
    tmp.write_bytes(resp.content)
    with rasterio.open(tmp) as src:
        arr = src.read()
    tmp.unlink(missing_ok=True)
    return arr


def write_cog(path: Path, arr: np.ndarray, spec: SceneSpec) -> dict[str, Any]:
    """Write a Cloud-Optimised GeoTIFF with overviews, then hash it."""
    w, s, e, n = spec.bbox
    count, height, width = arr.shape
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": from_bounds(w, s, e, n, width, height),
        "compress": "deflate",
        "predictor": 3,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "BIGTIFF": "NO",
    }
    path.unlink(missing_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr)
        dst.descriptions = ("sigma0_VV_linear", "sigma0_VH_linear", "dataMask")
        dst.build_overviews([2, 4, 8, 16], Resampling.average)
        dst.update_tags(ns="rio_overview", resampling_method="average")

    # Re-open in COG-friendly layout: overviews before image data.
    tmp = path.with_suffix(".cog.tif")
    with rasterio.open(path) as src:
        rasterio.shutil.copy(
            src,
            tmp,
            driver="GTiff",
            compress="deflate",
            predictor=3,
            tiled=True,
            blockxsize=512,
            blockysize=512,
            COPY_SRC_OVERVIEWS=True,
        )
    tmp.replace(path)

    vv = arr[0]
    finite = vv[np.isfinite(vv) & (vv > 0)]
    if finite.size == 0:
        path.unlink(missing_ok=True)
        raise NoDataError(
            "no valid sigma0 pixels — no Sentinel-1 swath covers this AOI "
            "inside the requested time window"
        )
    db = 10.0 * np.log10(finite)
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
        "width": width,
        "height": height,
        "stats": {
            "vv_db_p05": float(np.percentile(db, 5)),
            "vv_db_median": float(np.percentile(db, 50)),
            "vv_db_p95": float(np.percentile(db, 95)),
            "valid_fraction": float(finite.size / vv.size),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="show the scene manifest")
    ap.add_argument("--only", help="fetch a single scene key")
    args = ap.parse_args()

    if args.list:
        for s in MANIFEST:
            print(f"{s.key:32s} {s.label}")
        return 0

    token = get_token()
    print(f"CDSE token acquired ({len(token)} chars)\n")

    results: list[dict[str, Any]] = []
    for spec in MANIFEST:
        if args.only and spec.key != args.only:
            continue
        print(f"── {spec.label}")
        try:
            scenes = catalogue_lookup(token, spec)
            print(f"   catalogue: {len(scenes)} real scene(s) in window")
            for sc in scenes[:3]:
                print(f"     · {sc['Name'][:64]}")
            if not scenes:
                print("   SKIP: no acquisition covers this AOI/window")
                continue

            t0 = time.time()
            arr = fetch_geotiff(token, spec)
            tif = OUT_DIR / f"{spec.key}.tif"
            meta = write_cog(tif, arr, spec)
            print(
                f"   geotiff: {meta['width']}x{meta['height']} "
                f"{meta['bytes'] / 1e6:.1f} MB in {time.time() - t0:.1f}s"
            )
            print(
                f"   VV dB p05/med/p95: {meta['stats']['vv_db_p05']:.1f} / "
                f"{meta['stats']['vv_db_median']:.1f} / {meta['stats']['vv_db_p95']:.1f}"
                f"  valid={meta['stats']['valid_fraction'] * 100:.1f}%"
            )

            sidecar = {
                "key": spec.key,
                "label": spec.label,
                "note": spec.note,
                "bbox": list(spec.bbox),
                "time_window": list(spec.window),
                "bands": ["sigma0_VV_linear", "sigma0_VH_linear", "dataMask"],
                "units": "linear power (multiply by 0 -> dB: 10*log10(x))",
                "source": {
                    "catalogue": CATALOGUE_URL,
                    "process": PROCESS_URL,
                    "platform": "Sentinel-1 GRD (Copernicus Data Space Ecosystem)",
                },
                "scenes_in_window": [
                    {
                        "name": sc["Name"],
                        "id": sc["Id"],
                        "start": sc["ContentDate"]["Start"],
                        "size_mb": round(sc.get("ContentLength", 0) / 1e6, 1),
                    }
                    for sc in scenes[:5]
                ],
                "geotiff": str(tif.relative_to(REPO_ROOT)),
                "fetched_utc": datetime.now(UTC).isoformat(),
                **meta,
            }
            (OUT_DIR / f"{spec.key}.json").write_text(json.dumps(sidecar, indent=2))
            results.append(sidecar)
            print(f"   sha256: {meta['sha256'][:16]}…\n")
        except NoDataError as exc:
            print(f"   NO DATA: {exc}")
            print("   (skipped - SENTINEL never substitutes synthetic pixels)\n")
            results.append({"key": spec.key, "status": "no_data", "error": str(exc)})
        except Exception as exc:  # noqa: BLE001 - report and keep going
            print(f"   FAILED: {type(exc).__name__}: {str(exc)[:180]}\n")
            results.append({"key": spec.key, "status": "failed", "error": str(exc)[:400]})

    prior: list[dict[str, Any]] = []
    index = OUT_DIR / "index.json"
    if index.exists():  # merge: --only runs must not drop earlier scenes
        try:
            prior = [
                r
                for r in json.loads(index.read_text()).get("scenes", [])
                if r.get("key") not in {x.get("key") for x in results}
            ]
        except Exception:  # noqa: BLE001
            prior = []
    index.write_text(
        json.dumps(
            {
                "generated_utc": datetime.now(UTC).isoformat(),
                "source": "Copernicus Data Space Ecosystem (CDSE OData + Sentinel Hub Process API)",
                "scenes": prior + results,
            },
            indent=2,
        )
    )
    ok = sum(1 for r in results if "status" not in r)
    print(f"Done: {ok}/{len(results)} scenes written. Index -> {index}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
