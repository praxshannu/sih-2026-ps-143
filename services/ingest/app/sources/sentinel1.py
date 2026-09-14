"""ESA Copernicus Hub auto-fetch for Sentinel-1 GRD/EOC products."""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime, timedelta

import httpx
from loguru import logger

from ..models.schemas import Sentinel1Product, Sentinel1Result

# Indian EEZ bounding box
INDIA_EEZ_BBOX = (68.0, 5.0, 88.0, 25.0)

COPERNICUS_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
)
COPERNICUS_CATALOGUE_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"


class Sentinel1Fetcher:
    """Fetches Sentinel-1 GRD products from Copernicus Data Space."""

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        storage_path: str = "/data/sentinel1",
    ) -> None:
        self.client_id = client_id or os.getenv("COPERNICUS_CLIENT_ID", "")
        self.client_secret = client_secret or os.getenv("COPERNICUS_CLIENT_SECRET", "")
        self.storage_path = storage_path
        self._token: str | None = None
        self._token_expires: datetime | None = None

    async def _get_token(self) -> str:
        """Authenticate with Copernicus Data Space and obtain a bearer token."""
        if self._token and self._token_expires and datetime.now(UTC) < self._token_expires:
            return self._token

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                COPERNICUS_TOKEN_URL,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        token = data.get("access_token", "")
        if not token:
            raise ValueError("No access_token in Copernicus auth response")
        self._token = token
        self._token_expires = datetime.now(UTC) + timedelta(
            seconds=data.get("expires_in", 600) - 60
        )
        logger.info("Copernicus token acquired, expires in {}s", data.get("expires_in", 600))
        return token

    async def search_products(
        self,
        bbox: tuple[float, float, float, float] = INDIA_EEZ_BBOX,
        product_type: str = "GRD",
        max_results: int = 50,
        lookback_days: int = 6,
    ) -> list[Sentinel1Product]:
        """Query Copernicus OData catalogue for recent Sentinel-1 products.

        Parameters
        ----------
        bbox : (lon_min, lat_min, lon_max, lat_max)
        product_type : GRD, SLC, or OCN
        max_results : Maximum products to return
        lookback_days : How far back to search

        Returns
        -------
        List of Sentinel1Product metadata dicts.
        """
        lon_min, lat_min, lon_max, lat_max = bbox
        date_end = datetime.now(UTC)
        date_start = date_end - timedelta(days=lookback_days)

        start_iso = date_start.strftime("%Y-%m-%dT00:00:00.000Z")
        end_iso = date_end.strftime("%Y-%m-%dT23:59:59.999Z")
        date_filter = f"ContentDate/Start ge {start_iso} and ContentDate/Start le {end_iso}"
        footprint_filter = (
            "OData.CSC.Intersects(area=geography'SRID=4326;POLYGON(("
            f"{lon_min} {lat_min},{lon_max} {lat_min},{lon_max} {lat_max},"
            f"{lon_min} {lat_max},{lon_min} {lat_min}))')"
        )
        collection_filter = (
            "Collection/Name eq 'SENTINEL-1' and "
            "Attributes/OData.CSC.StringAttribute/any(att:att/Name eq 'productType' and "
            f"att/OData.CSC.StringAttribute/Value eq '{product_type}')"
        )

        filter_str = f"{collection_filter} and {footprint_filter} and {date_filter}"

        params = {
            "$filter": filter_str,
            "$orderby": "ContentDate/Start desc",
            "$top": str(max_results),
            "$expand": "Attributes",
        }

        products: list[Sentinel1Product] = []

        async with httpx.AsyncClient(timeout=60) as client:
            logger.info(
                "Querying Copernicus catalogue: product_type={}, bbox={}, window={}d",
                product_type,
                bbox,
                lookback_days,
            )

            resp = await client.get(COPERNICUS_CATALOGUE_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

            for item in data.get("value", []):
                attrs = {a["Name"]: a.get("Value", "") for a in item.get("Attributes", [])}

                orbit_num = int(attrs.get("orbitNumber", "0") or "0")
                orbit_dir = attrs.get("orbitDirection", "DESCENDING").upper()
                polarization = attrs.get("polarisation", "VV").upper()
                footprint_wkt = attrs.get("footprint", "")

                product = Sentinel1Product(
                    product_id=item["Id"],
                    name=item["Name"],
                    acquisition_time=datetime.fromisoformat(
                        item["ContentDate"]["Start"].replace("Z", "+00:00")
                    ),
                    orbit_number=orbit_num,
                    orbit_direction=orbit_dir,  # type: ignore[arg-type]
                    polarization=polarization,  # type: ignore[arg-type]
                    product_type=product_type,
                    footprint=footprint_wkt,
                    download_url=f"https://catalogue.dataspace.copernicus.eu/odata/v1/Products({item['Id']})/$value",
                    file_size_bytes=item.get("ContentLength"),
                )
                products.append(product)

        logger.info("Found {} Sentinel-1 {} products", len(products), product_type)
        return products

    async def download_product(
        self,
        product: Sentinel1Product,
    ) -> str:
        """Download a Sentinel-1 product to local storage.

        Returns the local file path.
        """
        token = await self._get_token()

        product_dir = os.path.join(self.storage_path, product.product_id)
        os.makedirs(product_dir, exist_ok=True)

        out_path = os.path.join(product_dir, f"{product.name}.zip")

        if os.path.exists(out_path):
            logger.info("Product already downloaded: {}", product.name)
            return out_path

        logger.info("Downloading product: {} -> {}", product.name, out_path)

        async with httpx.AsyncClient(timeout=300) as client:
            async with client.stream(
                "GET",
                product.download_url,
                headers={"Authorization": f"Bearer {token}"},
            ) as resp:
                resp.raise_for_status()
                with open(out_path, "wb") as f:
                    async for chunk in resp.aiter_bytes(chunk_size=1024 * 1024):
                        f.write(chunk)

        product.status = "downloaded"
        logger.info("Downloaded: {} ({} bytes)", product.name, os.path.getsize(out_path))
        return out_path

    async def fetch(
        self,
        bbox: tuple[float, float, float, float] = INDIA_EEZ_BBOX,
        product_type: str = "GRD",
        max_results: int = 50,
        lookback_days: int = 6,
    ) -> Sentinel1Result:
        """Full fetch cycle: search + download new products.

        Returns Sentinel1Result with download metadata.
        """
        t0 = time.time()

        products = await self.search_products(
            bbox=bbox,
            product_type=product_type,
            max_results=max_results,
            lookback_days=lookback_days,
        )

        downloaded = 0
        skipped = 0
        product_results: list[Sentinel1Product] = []

        for product in products:
            product_dir = os.path.join(self.storage_path, product.product_id)
            if os.path.exists(os.path.join(product_dir, f"{product.name}.zip")):
                product.status = "cached"
                skipped += 1
            else:
                try:
                    await self.download_product(product)
                    downloaded += 1
                except Exception as e:
                    logger.error("Failed to download {}: {}", product.name, str(e))
                    product.status = "error"
            product_results.append(product)

        elapsed = time.time() - t0

        logger.info(
            "Sentinel-1 fetch complete: {} found, {} downloaded, {} skipped in {:.1f}s",
            len(products),
            downloaded,
            skipped,
            elapsed,
        )

        return Sentinel1Result(
            products_found=len(products),
            products_downloaded=downloaded,
            products_skipped=skipped,
            products=product_results,
            fetch_duration_seconds=round(elapsed, 2),
        )
