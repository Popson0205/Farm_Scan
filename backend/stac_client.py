"""
Searches and fetches Sentinel-2 L2A (surface reflectance) scenes from the
Microsoft Planetary Computer STAC catalog. No API key required.

Docs: https://planetarycomputer.microsoft.com/api/stac/v1
"""
from datetime import datetime
from typing import Optional

import numpy as np
import planetary_computer
import rasterio
from pystac_client import Client
from rasterio.mask import mask as rio_mask
from rasterio.warp import transform_geom

STAC_API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"

# Sentinel-2 L2A asset keys -> our band role names
ASSET_MAP = {
    "blue": "B02",
    "green": "B03",
    "red": "B04",
    "rededge": "B05",
    "nir": "B08",
    "swir1": "B11",
}


def search_scenes(geojson_geometry: dict, start_date: str, end_date: str,
                   max_cloud_cover: float = 30.0, limit: int = 12) -> list[dict]:
    """
    Search for Sentinel-2 scenes intersecting a farm boundary polygon.

    geojson_geometry: a GeoJSON Polygon/MultiPolygon geometry dict (the farm boundary)
    start_date / end_date: 'YYYY-MM-DD'
    Returns a list of lightweight scene summaries (not full imagery) for the user to pick from.
    """
    catalog = Client.open(STAC_API_URL)

    search = catalog.search(
        collections=[COLLECTION],
        intersects=geojson_geometry,
        datetime=f"{start_date}/{end_date}",
        query={"eo:cloud_cover": {"lt": max_cloud_cover}},
        limit=limit,
        sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}],
    )

    results = []
    for item in search.item_collection():
        signed = planetary_computer.sign(item)
        results.append({
            "scene_id": signed.id,
            "date": signed.properties.get("datetime"),
            "cloud_cover": signed.properties.get("eo:cloud_cover"),
            "platform": signed.properties.get("platform"),
            "thumbnail_url": signed.assets.get("rendered_preview", {}).href
                if signed.assets.get("rendered_preview") else None,
            "_href_cache": {role: signed.assets[key].href
                             for role, key in ASSET_MAP.items() if key in signed.assets},
        })
    return results


def fetch_and_clip_bands(scene_href_map: dict, boundary_geojson: dict,
                          bands_needed: list[str]) -> tuple[dict, dict]:
    """
    Download only the requested bands for one scene, clip each to the farm boundary,
    and return band arrays (float32, reflectance 0-1) plus the raster transform/crs metadata
    needed to georeference the output overlay.

    scene_href_map: {'red': 'https://...B04.tif', 'nir': 'https://...B08.tif', ...}
                     (this is the '_href_cache' value returned by search_scenes for the chosen scene)
    boundary_geojson: GeoJSON geometry in EPSG:4326
    bands_needed: e.g. ['red', 'nir'] as required by the chosen index
    """
    band_arrays = {}
    out_meta = None

    for role in bands_needed:
        href = scene_href_map.get(role)
        if href is None:
            raise ValueError(f"Scene is missing required band '{role}' ({ASSET_MAP.get(role)})")

        with rasterio.open(href) as src:
            geom_in_raster_crs = transform_geom("EPSG:4326", src.crs, boundary_geojson)
            clipped, transform = rio_mask(src, [geom_in_raster_crs], crop=True, nodata=0)
            arr = clipped[0].astype(np.float32) / 10000.0  # Sentinel-2 L2A scale factor
            arr[clipped[0] == 0] = np.nan
            band_arrays[role] = arr

            if out_meta is None:
                out_meta = {
                    "transform": transform,
                    "crs": src.crs,
                    "height": arr.shape[0],
                    "width": arr.shape[1],
                }

    # align shapes defensively (different bands can be at 10m/20m native res before COG overview matching)
    target_shape = out_meta["height"], out_meta["width"]
    for role, arr in band_arrays.items():
        if arr.shape != target_shape:
            raise ValueError(
                f"Band '{role}' shape {arr.shape} does not match reference shape {target_shape}; "
                "reproject/resample bands to a common grid before computing indices."
            )

    return band_arrays, out_meta
