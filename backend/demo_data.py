"""
Generates plausible synthetic Sentinel-2 reflectance bands for a farm boundary,
so the full pipeline (boundary -> bands -> index -> overlay -> stats) can be
exercised without a live connection to the satellite imagery provider.

Swap this out for stac_client.fetch_and_clip_bands() once deployed somewhere
with outbound access to planetarycomputer.microsoft.com.
"""
import numpy as np
from shapely.geometry import shape
from rasterio.transform import from_bounds
from rasterio.features import geometry_mask


def synthetic_bands_for_boundary(boundary_geojson: dict, resolution: int = 128, seed: int = 0):
    """
    Builds a raster grid covering the boundary's bounding box, rasterizes the
    polygon as a mask, and fills in spatially-correlated fake reflectance values
    for blue/green/red/rededge/nir/swir1 so realistic-looking indices come out.
    """
    geom = shape(boundary_geojson)
    minx, miny, maxx, maxy = geom.bounds
    height = width = resolution

    transform = from_bounds(minx, miny, maxx, maxy, width, height)
    field_mask = geometry_mask([boundary_geojson], out_shape=(height, width),
                                transform=transform, invert=True)

    rng = np.random.default_rng(seed)

    # Smooth spatial noise field (simulates patchy crop vigor/moisture) via low-res upsampling
    def smooth_field(low_res=8, base=0.5, spread=0.2):
        low = rng.uniform(base - spread, base + spread, size=(low_res, low_res))
        ys = np.linspace(0, low_res - 1, height)
        xs = np.linspace(0, low_res - 1, width)
        y0 = np.clip(ys.astype(int), 0, low_res - 1)
        x0 = np.clip(xs.astype(int), 0, low_res - 1)
        return low[np.ix_(y0, x0)]

    vigor = smooth_field(base=0.6, spread=0.25)   # drives NIR/red relationship
    moisture = smooth_field(base=0.4, spread=0.2)  # drives swir/green relationship

    red = np.clip(0.12 + (1 - vigor) * 0.10 + rng.normal(0, 0.01, (height, width)), 0.02, 0.4)
    nir = np.clip(0.20 + vigor * 0.45 + rng.normal(0, 0.02, (height, width)), 0.05, 0.7)
    rededge = np.clip((red + nir) / 2 + vigor * 0.05, 0.05, 0.6)
    green = np.clip(0.10 + (1 - moisture) * 0.05 + rng.normal(0, 0.01, (height, width)), 0.02, 0.3)
    swir1 = np.clip(0.30 - moisture * 0.20 + rng.normal(0, 0.02, (height, width)), 0.05, 0.5)
    blue = np.clip(0.08 + rng.normal(0, 0.008, (height, width)), 0.01, 0.2)

    bands = {"red": red, "nir": nir, "rededge": rededge, "green": green, "swir1": swir1, "blue": blue}
    for arr in bands.values():
        arr[~field_mask] = np.nan

    out_meta = {"transform": transform, "crs": "EPSG:4326", "height": height, "width": width}
    return bands, out_meta, field_mask
