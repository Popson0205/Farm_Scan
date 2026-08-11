"""
Ingests a user-uploaded drone orthomosaic (the stitched GeoTIFF output of
Pix4D / Agisoft Metashape / DroneDeploy / OpenDroneMap) so the same
index-calculation and plot-extraction pipeline used for satellite imagery
can also run on centimeter-resolution drone data — which is required for
plant counts and fine-grained (<1m) management zones.

This does NOT do the stitching itself (that's what Agisoft Metashape/ODM do
from raw drone photos) — it consumes their already-stitched output.
"""
import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
from rasterio.warp import transform_geom, calculate_default_transform, reproject, Resampling


def inspect_orthomosaic(file_path: str) -> dict:
    """Returns basic metadata so the frontend can ask the user to confirm band mapping
    (drone sensors vary: some are RGB-only, some RGB+NIR, some 5-band multispectral)."""
    with rasterio.open(file_path) as src:
        return {
            "band_count": src.count,
            "width": src.width,
            "height": src.height,
            "crs": str(src.crs),
            "resolution_m": (abs(src.transform.a), abs(src.transform.e)),
            "bounds": list(src.bounds),
            "dtype": str(src.dtypes[0]),
        }


def generate_preview(file_path: str, max_dim: int = 1400) -> dict:
    """
    Renders a downsampled RGB PNG preview of the uploaded orthomosaic plus its
    footprint in EPSG:4326, so the frontend can drop it onto the Leaflet map as
    an image overlay right after upload — otherwise the user has no visual
    confirmation the file loaded correctly until they run plant count/plot
    extraction.
    """
    import io
    import base64
    from PIL import Image
    from rasterio.warp import transform_bounds

    with rasterio.open(file_path) as src:
        band_count = min(src.count, 3)
        scale = min(1.0, max_dim / max(src.width, src.height))
        out_w = max(1, int(src.width * scale))
        out_h = max(1, int(src.height * scale))

        data = src.read(
            list(range(1, band_count + 1)),
            out_shape=(band_count, out_h, out_w),
            resampling=rasterio.enums.Resampling.average,
        )

        if data.dtype != np.uint8:
            lo, hi = np.nanpercentile(data, 1), np.nanpercentile(data, 99)
            data = np.clip((data.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1) * 255
            data = data.astype(np.uint8)

        if band_count == 1:
            rgb = np.repeat(data[0][..., None], 3, axis=2)
        else:
            rgb = np.transpose(data[:3], (1, 2, 0))

        img = Image.fromarray(rgb, mode="RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_b64 = base64.b64encode(buf.getvalue()).decode()

        west, south, east, north = transform_bounds(src.crs, "EPSG:4326", *src.bounds)

    return {
        "preview_png_base64": png_b64,
        "bounds_wgs84": [west, south, east, north],  # [minLon, minLat, maxLon, maxLat]
    }


def load_bands_clipped(file_path: str, boundary_geojson: dict, band_map: dict) -> tuple[dict, dict]:
    """
    band_map: which file band index (1-indexed, as in the GeoTIFF) maps to which
    role, e.g. {"red": 1, "green": 2, "blue": 3, "nir": 4} for an RGB+NIR sensor.
    Missing roles (e.g. no NIR on an RGB-only drone) simply aren't returned —
    the caller should only request indices whose required bands are present.

    Returns (band_arrays dict of role->float32 array, meta dict with transform/crs).
    """
    band_arrays = {}
    out_meta = None

    with rasterio.open(file_path) as src:
        geom_in_raster_crs = transform_geom("EPSG:4326", src.crs, boundary_geojson)
        band_indices = list(band_map.values())
        clipped, transform = rio_mask(src, [geom_in_raster_crs], crop=True,
                                       indexes=band_indices, nodata=0)

        dtype_max = 255.0 if src.dtypes[0] == "uint8" else 65535.0

        for i, (role, band_idx) in enumerate(band_map.items()):
            arr = clipped[i].astype(np.float32) / dtype_max
            arr[clipped[i] == 0] = np.nan
            band_arrays[role] = arr

        out_meta = {"transform": transform, "crs": src.crs,
                    "height": clipped.shape[1], "width": clipped.shape[2]}

    return band_arrays, out_meta


def load_rgb_for_plant_count(file_path: str, boundary_geojson: dict = None) -> tuple[np.ndarray, dict]:
    """Loads the first 3 bands as an HxWx3 uint8 array for plant_count.py,
    optionally clipped to a boundary. Assumes band order is R,G,B (true for
    almost all drone RGB/RGB+NIR sensor outputs)."""
    with rasterio.open(file_path) as src:
        if boundary_geojson is not None:
            geom_in_raster_crs = transform_geom("EPSG:4326", src.crs, boundary_geojson)
            clipped, transform = rio_mask(src, [geom_in_raster_crs], crop=True, indexes=[1, 2, 3])
        else:
            clipped = src.read([1, 2, 3])
            transform = src.transform

        rgb = np.transpose(clipped, (1, 2, 0))
        if rgb.dtype != np.uint8:
            rgb = cv2_safe_normalize(rgb)

        return rgb, {"transform": transform, "crs": src.crs}


def cv2_safe_normalize(arr: np.ndarray) -> np.ndarray:
    """Normalizes non-8bit imagery (e.g. 16-bit drone sensor output) to uint8 for
    the RGB-based plant detector, without requiring OpenCV at import time here."""
    arr = arr.astype(np.float32)
    lo, hi = np.nanpercentile(arr, 1), np.nanpercentile(arr, 99)
    arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1) * 255
    return arr.astype(np.uint8)
