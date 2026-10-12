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
    for geom_dict, value in rio_shapes(zone_uint, mask=mask, transform=transform):
        polygons_by_zone.setdefault(int(value), []).append(shape(geom_dict))
    # true pixel count per zone (previously this counted polygon fragments, not pixels)
    pixel_counts = {z: int((zone_array == z).sum()) for z in polygons_by_zone}

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


# ---------------------------------------------------------------------------
# Variable-rate plan
# ---------------------------------------------------------------------------

STRATEGIES = {
    # Low-vigor zones get MORE input. Typical when the shortfall is nutrient-driven
    # (e.g. nitrogen top-dress where NDRE is low) and the zone can still respond.
    "compensate": -1.0,
    # High-vigor zones get MORE input. Typical when the best-performing areas have the
    # highest yield potential and weak zones are limited by something inputs won't fix
    # (waterlogging, compaction, shallow soil).
    "capitalize": +1.0,
}


def build_rate_plan(zone_polygons: list[dict], px_area_m2: float, base_rate: float,
                    unit: str = "kg/ha", strategy: str = "compensate",
                    spread: float = 0.2) -> dict:
    """
    Turns classified zones into an actionable rate plan.

    base_rate: the agronomist's flat-rate recommendation for the input (e.g. 90 kg N/ha).
    spread:    maximum fractional change applied to the extreme zones (0.2 = +/-20%).
    strategy:  'compensate' or 'capitalize' (see STRATEGIES). Which one is right depends
               on WHY the weak zones are weak, so it is a choice the agronomist makes,
               not something the satellite can decide.

    Each zone gets rate, area_ha and product_total. The plan compares total product
    against applying base_rate everywhere. The difference can be positive or negative:
    the plan redistributes input, and any saving or yield gain must be validated with a
    strip trial before it is relied on.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {list(STRATEGIES)}")
    if base_rate <= 0:
        raise ValueError("base_rate must be greater than 0")
    if not 0 <= spread <= 0.5:
        raise ValueError("spread must be between 0 and 0.5")

    n = len(zone_polygons)
    direction = STRATEGIES[strategy]
    total_area_ha = 0.0
    for z in zone_polygons:
        z["area_ha"] = round(z["pixel_count"] * px_area_m2 / 10000.0, 4)
        total_area_ha += z["area_ha"]

    variable_total = 0.0
    for z in zone_polygons:
        pos = (2.0 * z["zone"] / (n - 1) - 1.0) if n > 1 else 0.0   # -1 (weakest) .. +1 (strongest)
        z["rate"] = round(base_rate * (1.0 + direction * spread * pos), 2)
        z["unit"] = unit
        z["product_total"] = round(z["rate"] * z["area_ha"], 2)
        variable_total += z["product_total"]

    flat_total = base_rate * total_area_ha
    return {
        "strategy": strategy, "base_rate": base_rate, "unit": unit, "spread": spread,
        "total_area_ha": round(total_area_ha, 3),
        "flat_rate_total": round(flat_total, 2),
        "variable_rate_total": round(variable_total, 2),
        "difference": round(variable_total - flat_total, 2),
        "difference_pct": round(100.0 * (variable_total - flat_total) / flat_total, 2) if flat_total else None,
        "note": ("Rates redistribute input across zones. Validate against a flat-rate strip "
                 "before relying on any saving or yield gain."),
    }
