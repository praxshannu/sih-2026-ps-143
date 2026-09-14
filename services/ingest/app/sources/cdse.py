"""Copernicus Data Space Ecosystem client for the SENTINEL archive browser.

Two upstreams are used, and the split is deliberate:

* **CDSE OData catalogue** (`catalogue.dataspace.copernicus.eu`) is the authority
  for *what exists* — product name, acquisition time, orbit, size, footprint.
* **CDSE Sentinel Hub Process API** (`sh.dataspace.copernicus.eu`) renders
  calibrated sigma0 for an arbitrary AOI server-side. This is the same engine the
  Copernicus Browser runs on, and it returns a few MB instead of a ~2 GB .SAFE zip.

Three CDSE quirks encoded here, each found by testing against the live service:

1. The OData parser rejects the ``datetime'...'`` literal prefix that SciHub-era
   examples use. Bare ISO literals are required or the query 400s.
2. ``OrbitNumber`` is not a valid ``$select`` field and 400s the whole request.
3. The download endpoint rejects client-credentials tokens ("Token audience not
   allowed"), so bulk product download is not attempted at all.

No synthetic fallback exists in this module. If an upstream returns nothing, the
caller receives an empty result or an exception — never invented pixels.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any

import httpx
from loguru import logger

TOKEN_URL = os.getenv(
    "CDSE_TOKEN_URL",
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
)
CATALOGUE_URL = os.getenv(
    "CDSE_CATALOG_URL", "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
)
PROCESS_URL = os.getenv("CDSE_PROCESS_URL", "https://sh.dataspace.copernicus.eu/api/v1/process")

# Fields CDSE actually accepts in $select. OrbitNumber is not among them.
CATALOGUE_SELECT = "Id,Name,ContentDate,ContentLength,GeoFootprint,Online"

# False-colour quicklook: (R,G,B) = (VV, VH, VV-VH). Oil slicks read as dark with
# a cyan cast against grey sea — the convention SAR analysts actually read.
QUICKLOOK_EVALSCRIPT = """
//VERSION=3
function setup() {
  return { input: [{ bands: ["VV", "VH", "dataMask"] }],
           output: { bands: 4, sampleType: "UINT8" } };
}
let lo = -32.0, hi = 2.0;
function stretch(v) {
  let x = 10.0 * Math.log(Math.max(v, 1e-6)) / Math.LN10;
  return Math.max(0.0, Math.min(1.0, (x - lo) / (hi - lo)));
}
function evaluatePixel(s) {
  let vv = stretch(s.VV), vh = stretch(s.VH);
  let ratio = Math.max(0.0, Math.min(1.0, vv - vh + 0.5));
  return [vv * 255, vh * 255, ratio * 255, s.dataMask * 255];
}
"""

# Analysis-ready stack: linear sigma0 VV/VH plus a validity mask.
GEOTIFF_EVALSCRIPT = """
//VERSION=3
function setup() {
  return { input: [{ bands: ["VV", "VH", "dataMask"] }],
           output: { bands: 3, sampleType: "FLOAT32" } };
}
function evaluatePixel(s) { return [s.VV, s.VH, s.dataMask]; }
"""


@dataclass
class Scene:
    """One catalogue entry, with its real footprint geometry."""

    id: str
    name: str
    start: str
    size_mb: float
    footprint: dict[str, Any] | None = None
    platform: str = ""
    mode: str = ""
    product_type: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "start": self.start,
            "size_mb": round(self.size_mb, 1),
            "footprint": self.footprint,
            "platform": self.platform,
            "mode": self.mode,
            "product_type": self.product_type,
        }


@dataclass
class CdseArchiveClient:
    """Async client for CDSE catalogue search and Sentinel Hub rendering."""

    client_id: str = field(default_factory=lambda: os.getenv("CDSE_CLIENT_ID", ""))
    client_secret: str = field(default_factory=lambda: os.getenv("CDSE_CLIENT_SECRET", ""))
    timeout: float = 120.0

    _token: str | None = field(default=None, init=False, repr=False)
    _token_expiry: float = field(default=0.0, init=False, repr=False)

    @property
    def configured(self) -> bool:
        return bool(self.client_id and self.client_secret)

    async def _get_token(self) -> str:
        """OAuth2 client-credentials token, cached until 60 s before expiry."""
        if self._token and time.time() < self._token_expiry:
            return self._token
        if not self.configured:
            raise RuntimeError(
                "CDSE_CLIENT_ID / CDSE_CLIENT_SECRET are not set — cannot query Copernicus"
            )
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                TOKEN_URL,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                },
            )
            resp.raise_for_status()
            payload = resp.json()
        self._token = str(payload["access_token"])
        # CDSE tokens live 1800 s; refresh a minute early.
        self._token_expiry = time.time() + float(payload.get("expires_in", 1800)) - 60
        logger.debug("CDSE token acquired, valid {}s", int(self._token_expiry - time.time()))
        return self._token

    @staticmethod
    def _build_filter(
        bbox: tuple[float, float, float, float],
        start: str,
        end: str,
        product_type: str | None,
        platform: str | None,
        mode: str | None,
    ) -> str:
        w, s, e, n = bbox
        ring = f"{w} {s}, {e} {s}, {e} {n}, {w} {n}, {w} {s}"
        parts = [
            "Collection/Name eq 'SENTINEL-1'",
            f"OData.CSC.Intersects(area=geography'SRID=4326;POLYGON(({ring}))')",
            f"ContentDate/Start gt {start}",
            f"ContentDate/Start lt {end}",
        ]
        if product_type:
            parts.append(
                "Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType'"
                f" and att/OData.CSC.StringAttribute/Value eq '{product_type}')"
            )
        if platform and platform.upper() != "ALL":
            parts.append(f"startswith(Name,'{platform.upper()}')")
        if mode and mode.upper() != "ALL":
            parts.append(f"contains(Name,'_{mode.upper()}_')")
        return " and ".join(parts)

    @staticmethod
    def _parse_scene(raw: dict[str, Any]) -> Scene:
        name = raw.get("Name", "")
        # S1A_IW_GRDH_1SDV_20200810... -> platform S1A, mode IW, type GRDH
        bits = name.split("_")
        return Scene(
            id=raw.get("Id", ""),
            name=name,
            start=(raw.get("ContentDate") or {}).get("Start", ""),
            size_mb=float(raw.get("ContentLength", 0) or 0) / 1e6,
            footprint=raw.get("GeoFootprint"),
            platform=bits[0] if bits else "",
            mode=bits[1] if len(bits) > 1 else "",
            product_type=bits[2] if len(bits) > 2 else "",
        )

    async def search(
        self,
        bbox: tuple[float, float, float, float],
        start: str,
        end: str,
        product_type: str | None = "GRD",
        platform: str | None = None,
        mode: str | None = None,
        top: int = 50,
    ) -> list[Scene]:
        """Search the real catalogue. Returns [] when nothing covers the AOI."""
        token = await self._get_token()
        params = {
            "$filter": self._build_filter(bbox, start, end, product_type, platform, mode),
            "$orderby": "ContentDate/Start desc",
            "$top": str(max(1, min(int(top), 100))),
            "$select": CATALOGUE_SELECT,
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(
                CATALOGUE_URL, params=params, headers={"Authorization": f"Bearer {token}"}
            )
            resp.raise_for_status()
            raw = resp.json()
        scenes = [self._parse_scene(v) for v in raw.get("value", [])]
        logger.info(
            "CDSE search bbox={} {}..{} -> {} scene(s)", bbox, start[:10], end[:10], len(scenes)
        )
        return scenes

    async def render(
        self,
        bbox: tuple[float, float, float, float],
        start: str,
        end: str,
        width: int,
        height: int,
        evalscript: str,
        fmt: str = "image/tiff",
        mosaicking_order: str = "mostRecent",
    ) -> bytes:
        """Render calibrated sigma0 for an AOI via the Sentinel Hub Process API."""
        token = await self._get_token()
        w, s, e, n = bbox
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
                            "timeRange": {"from": start, "to": end},
                            "mosaickingOrder": mosaicking_order,
                            "resolution": "HIGH",
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
                "width": width,
                "height": height,
                "responses": [{"identifier": "default", "format": {"type": fmt}}],
            },
            "evalscript": evalscript,
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                PROCESS_URL,
                json=payload,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                    "Accept": fmt,
                },
            )
            resp.raise_for_status()
            return resp.content

    async def quicklook(
        self,
        bbox: tuple[float, float, float, float],
        start: str,
        end: str,
        size: int = 512,
    ) -> bytes:
        """False-colour PNG preview for the browser map."""
        return await self.render(
            bbox, start, end, size, size, QUICKLOOK_EVALSCRIPT, fmt="image/png"
        )

    async def geotiff(
        self,
        bbox: tuple[float, float, float, float],
        start: str,
        end: str,
        size: int = 2048,
    ) -> bytes:
        """Analysis-ready float32 GeoTIFF (sigma0 VV, sigma0 VH, dataMask)."""
        return await self.render(
            bbox, start, end, size, size, GEOTIFF_EVALSCRIPT, fmt="image/tiff"
        )


_client: CdseArchiveClient | None = None


def get_cdse_client() -> CdseArchiveClient:
    """Process-wide client so the OAuth token is reused across requests."""
    global _client
    if _client is None:
        _client = CdseArchiveClient()
    return _client
