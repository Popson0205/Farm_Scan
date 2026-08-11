"""
Turns a continuous index raster (e.g. NDVI) into discrete management zones —
the input variable-rate spreaders/sprayers need — by clustering pixel values
into N classes, then polygonizing each class into zone geometries with a
recommended application rate per zone.
"""
import numpy as np
from sklearn.cluster import KMeans
from rasterio.features import shapes as rio_shapes
from shapely.geometry import shape, mapping
from shapely.ops import unary_union


def classify_zones(index_array: np.ndarray, n_zones: int = 3, seed: int = 42) -> np.ndarray:
    """
    K-Means clusters valid (non-NaN) pixel values into n_zones classes, ordered
    so that zone 0 = lowest index values (e.g. poorest vegetation) through
    zone n-1 = highest. Returns an int array same shape as input, with -1 for
    NaN/masked pixels.
    """
    valid_mask = ~np.isnan(index_array)
    flat_valid = index_array[valid_mask].reshape(-1, 1)

    if flat_valid.shape[0] < n_zones:
        raise ValueError(f"Not enough valid pixels ({flat_valid.shape[0]}) to form {n_zones} zones")

    km = KMeans(n_clusters=n_zones, random_state=seed, n_init=10)
    labels = km.fit_predict(flat_valid)

    # reorder cluster labels by ascending centroid value so zone numbering is meaningful
    centroid_order = np.argsort(km.cluster_centers_.flatten())
    remap = {old: new for new, old in enumerate(centroid_order)}
    labels_remapped = np.array([remap[l] for l in labels])

    zone_array = np.full(index_array.shape, -1, dtype=np.int16)
    zone_array[valid_mask] = labels_remapped
    return zone_array


# default application-rate suggestions per zone tier — placeholder logic;
# real deployments should let the agronomist set rates per crop/input
DEFAULT_RATE_TABLE_3 = {0: "High (low vigor -> more input)", 1: "Standard", 2: "Low (already vigorous)"}
DEFAULT_RATE_TABLE_5 = {
    0: "Very high", 1: "High", 2: "Standard", 3: "Low", 4: "Very low",
}


def zones_to_polygons(zone_array: np.ndarray, transform, crs, n_zones: int) -> list[dict]:
    """
    Polygonizes the zone raster into one (possibly multi-part) geometry per
    zone class, with pixel-count and suggested rate label as attributes.
    """
    rate_table = DEFAULT_RATE_TABLE_5 if n_zones > 3 else DEFAULT_RATE_TABLE_3
    if n_zones not in (3, 5):
        rate_table = {i: f"Zone {i}" for i in range(n_zones)}

    zone_uint = np.where(zone_array < 0, 255, zone_array).astype(np.uint8)  # 255 = nodata sentinel
    mask = zone_array >= 0

    polygons_by_zone = {}
    pixel_counts = {}
    for geom_dict, value in rio_shapes(zone_uint, mask=mask, transform=transform):
        z = int(value)
        polygons_by_zone.setdefault(z, []).append(shape(geom_dict))
        pixel_counts[z] = pixel_counts.get(z, 0) + 1

    results = []
    for z, geoms in sorted(polygons_by_zone.items()):
        merged = unary_union(geoms)
        results.append({
            "zone": z,
            "geometry": mapping(merged),
            "pixel_count": pixel_counts[z],
            "suggested_rate": rate_table.get(z, f"Zone {z}"),
        })
    return results
