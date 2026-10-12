"""
Searches and fetches Sentinel-2 L2A (surface reflectance) scenes from the
Microsoft Planetary Computer STAC catalog. No API key required.

Docs: https://planetarycomputer.microsoft.com/api/stac/v1

Accuracy notes (these matter for agronomic decisions):
  * Cloud / shadow masking: every scene ships a Scene Classification Layer
    (SCL). Pixels classed as cloud, cirrus, cloud shadow, saturated, snow or
    no-data are set to NaN so they never contaminate an index or a stat. This
    is essential in the Nigerian rainy season, when "30% cloud cover" scenes
    often have haze/shadow over the field itself.
  * Common grid: bands come at 10 m (B02/B03/B04/B08) and 20 m (B05/B11, SCL).
    Everything is warped onto the grid of a 10 m reference band, so indices
    that mix resolutions (NDRE, NDMI) work.
  * Reflectance offset: scenes processed with baseline >= 04.00 (since
    Jan 2022) carry a +1000 DN offset. Ignoring it biases every index
    (NDVI reads too low). The offset is taken from the STAC asset metadata
    (raster:bands) and falls back to the processing-baseline rule.
"""
from typing import Optional

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.mask import mask as rio_mask
from rasterio.vrt import WarpedVRT
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
    "scl": "SCL",
}

# 10 m bands, preferred as the reference grid
TEN_METER_ROLES = ("red", "nir", "green", "blue")

# SCL classes: 0 no data, 1 saturated/defective, 2 dark area pixels,
# 3 cloud shadow, 4 vegetation, 5 not vegetated, 6 water, 7 unclassified,
# 8 cloud (medium prob), 9 cloud (high prob), 10 thin cirrus, 11 snow/ice.
# Keep only classes that are usable land/water observations.
SCL_VALID_CLASSES = (4, 5, 6, 7)

# L2A processing baseline 04.00 introduced a +1000 DN offset (-0.1 reflectance)
_OFFSET_BASELINE = "04.00"
_DEFAULT_OFFSET = -0.1


def scl_valid_mask(scl: np.ndarray) -> np.ndarray:
    """True where the Scene Classification Layer says the pixel is a clear,
    usable observation (not cloud, cirrus, shadow, saturated, snow, no-data)."""
    return np.isin(scl, SCL_VALID_CLASSES)


def _reflectance_offset(item) -> float:
    """Reflectance offset for a scene: from STAC raster:bands metadata when
    present, otherwise from the processing baseline."""
    for key in ("B04", "B08"):
        asset = item.assets.get(key)
        if asset is not None:
            bands = (asset.extra_fields or {}).get("raster:bands") or []
            if bands and "offset" in bands[0]:
                return float(bands[0]["offset"])
    baseline = str(item.properties.get("s2:processing_baseline", ""))
    if baseline and baseline >= _OFFSET_BASELINE:
        return _DEFAULT_OFFSET
    return 0.0


def search_scenes(geojson_geometry: dict, start_date: str, end_date: str,
                   max_cloud_cover: float = 30.0, limit: int = 12) -> list[dict]:
    """
    Search for Sentinel-2 scenes intersecting a farm boundary polygon.

    geojson_geometry: a GeoJSON Polygon/MultiPolygon geometry dict (the farm boundary)
    start_date / end_date: 'YYYY-MM-DD'
    Returns a list of lightweight scene summaries (not full imagery) for the user to pick from.
    """
    import planetary_computer
    from pystac_client import Client

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
            "_reflectance_offset": _reflectance_offset(signed),
        })
    return results


def _read_on_grid(href: str, crs, transform, width: int, height: int,
                  resampling: Resampling) -> np.ndarray:
    """Read band 1 of `href` resampled onto the reference grid."""
    with rasterio.open(href) as src:
        with WarpedVRT(src, crs=crs, transform=transform, width=width, height=height,
                       resampling=resampling, nodata=0) as vrt:
            return vrt.read(1)


def fetch_and_clip_bands(scene_href_map: dict, boundary_geojson: dict,
                          bands_needed: list[str], reflectance_offset: float = 0.0,
                          apply_cloud_mask: bool = True) -> tuple[dict, dict]:
    """
    Download only the requested bands for one scene, clip to the farm boundary,
    put every band on one common 10 m grid, apply the reflectance offset and
    (by default) the SCL cloud/shadow mask.

    scene_href_map: {'red': 'https://...B04.tif', 'nir': ..., 'scl': ...}
                     (the '_href_cache' value returned by search_scenes)
    boundary_geojson: GeoJSON geometry in EPSG:4326
    reflectance_offset: from search_scenes()['_reflectance_offset']
    Returns (band_arrays float32 reflectance 0-1 with NaN where invalid, out_meta).

    out_meta carries transform/crs/height/width plus:
      field_pixels        pixels inside the boundary polygon
      valid_pixels        of those, pixels that are cloud-free and have data
      valid_fraction      valid_pixels / field_pixels  (data quality signal)
    """
    for role in bands_needed:
        if scene_href_map.get(role) is None:
            raise ValueError(f"Scene is missing required band '{role}' ({ASSET_MAP.get(role)})")

    ref_role = next((r for r in bands_needed if r in TEN_METER_ROLES), None)
    if ref_role is None:
        ref_role = next((r for r in TEN_METER_ROLES if scene_href_map.get(r)), None)
        if ref_role is None:
            raise ValueError("Scene has no 10 m band to use as the reference grid")

    # reference band: clip to the polygon, which defines the output grid
    with rasterio.open(scene_href_map[ref_role]) as src:
        geom_in_raster_crs = transform_geom("EPSG:4326", src.crs, boundary_geojson)
        clipped, transform = rio_mask(src, [geom_in_raster_crs], crop=True, nodata=0)
        ref_dn = clipped[0]
        crs = src.crs
    height, width = ref_dn.shape

    field_mask = geometry_mask([geom_in_raster_crs], out_shape=(height, width),
                               transform=transform, invert=True)

    raw = {ref_role: ref_dn}
    for role in bands_needed:
        if role in raw:
            continue
        raw[role] = _read_on_grid(scene_href_map[role], crs, transform, width, height,
                                  Resampling.bilinear)

    valid = field_mask.copy()
    if apply_cloud_mask and scene_href_map.get("scl"):
        scl = _read_on_grid(scene_href_map["scl"], crs, transform, width, height,
                            Resampling.nearest)
        valid &= scl_valid_mask(scl)

    band_arrays = {}
    for role, dn in raw.items():
        arr = dn.astype(np.float32) / 10000.0 + reflectance_offset
        arr = np.clip(arr, 0.0, None)          # offset can push dark pixels slightly negative
        arr[dn == 0] = np.nan                  # true no-data
        arr[~valid] = np.nan                   # outside the field, or cloud/shadow
        band_arrays[role] = arr

    field_pixels = int(field_mask.sum())
    valid_pixels = int((valid & ~np.isnan(band_arrays[bands_needed[0]])).sum())
    out_meta = {
        "transform": transform, "crs": crs, "height": height, "width": width,
        "field_pixels": field_pixels, "valid_pixels": valid_pixels,
        "valid_fraction": (valid_pixels / field_pixels) if field_pixels else 0.0,
        "cloud_masked": bool(apply_cloud_mask and scene_href_map.get("scl")),
    }
    return band_arrays, out_meta
