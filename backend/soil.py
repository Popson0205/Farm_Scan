"""
Soil constituents for a farm boundary from the Harmonized World Soil Database
v2.0 (FAO / IIASA).

HWSD is NOT a web API. It ships as
  * a ~1 km raster (HWSD2.bil / .tif) whose pixel value is a Soil Mapping Unit id, and
  * an attribute table (HWSD2_LAYERS in the .mdb) holding, per mapping unit, one or
    more soil units (with a % SHARE of the unit) x 7 depth layers (D1..D7).
So FarmScan hosts both locally (one-time setup: scripts/build_hwsd_sqlite.py) and
queries them offline -- fast, free, and no rate limits.

Lookup path
  boundary polygon -> raster pixels under it -> SMU ids (+ pixel counts)
    -> HWSD2_LAYERS rows -> weight by soil-unit SHARE and pixel count
    -> per-depth averages -> topsoil (0-30 cm) / subsoil (30-100 cm)

Accuracy notes (important for decisions)
  * ~1 km pixels (~100 ha). A typical farm sits inside ONE pixel, so the result
    is the regional soil association, not your field. It tells you "what soils
    are typical here", good for screening crops and planning, NOT for fertilizer
    rates. Confirm with a lab soil test (the 'soil_test_needed' flag says so).
  * Values are modelled/harmonised from many national surveys; quality varies by
    country. HWSD error statistics live in the WISE30sec database.
  * Missing values (NULL / negative) are ignored and the weights re-normalised.

Setup
  HWSD2_RASTER  path to HWSD2.bil/.tif   (default backend/data/hwsd/HWSD2.bil)
  HWSD2_DB      path to hwsd2.sqlite     (default backend/data/hwsd/hwsd2.sqlite)
"""
import os
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent / "data" / "hwsd"

# depth layers in HWSD v2.0 (cm)
LAYER_DEPTHS = {"D1": (0, 20), "D2": (20, 40), "D3": (40, 60), "D4": (60, 80),
                "D5": (80, 100), "D6": (100, 150), "D7": (150, 200)}

# numeric attributes we aggregate (column name in HWSD2_LAYERS -> our key)
NUMERIC_PROPS = {
    "SAND": "sand_pct", "SILT": "silt_pct", "CLAY": "clay_pct",
    "COARSE": "coarse_frag_pct",           # vol % coarse fragments
    "BULK": "bulk_density", "ORG_CARBON": "organic_carbon_pct", "PH_WATER": "ph_water",
    "TOTAL_N": "total_n_g_kg", "CN_RATIO": "cn_ratio",
    "CEC_SOIL": "cec_cmol_kg", "CEC_CLAY": "cec_clay_cmol_kg", "BSAT": "base_sat_pct",
    "TEB": "total_exch_bases_cmol_kg", "ESP": "exch_sodium_pct",
    "ELEC_COND": "ec_ds_m", "TCARBON_EQ": "caco3_pct", "GYPSUM": "gypsum_pct",
    "ALUM_SAT": "al_sat_pct", "AWC": "awc",
}

DRAINAGE_LABELS = {1: "excessively drained", 2: "somewhat excessively drained", 3: "well drained",
                   4: "moderately well drained", 5: "imperfectly drained", 6: "poorly drained",
                   7: "very poorly drained"}


class SoilDataUnavailable(RuntimeError):
    """HWSD files are not installed (or the boundary lies outside the raster)."""


# ---------------------------------------------------------------------------
# Paths / availability (read env at call time so tests and deployments can override)
# ---------------------------------------------------------------------------

def raster_path() -> Path:
    return Path(os.environ.get("HWSD2_RASTER", DATA_DIR / "HWSD2.bil"))


def db_path() -> Path:
    return Path(os.environ.get("HWSD2_DB", DATA_DIR / "hwsd2.sqlite"))


def status() -> dict:
    r, d = raster_path(), db_path()
    return {"raster": str(r), "raster_found": r.exists(), "database": str(d), "database_found": d.exists(),
            "ready": r.exists() and d.exists()}


# ---------------------------------------------------------------------------
# Raster sampling (rasterio imported lazily: only this function needs it)
# ---------------------------------------------------------------------------

def smu_histogram(boundary_geojson: dict) -> dict[int, int]:
    """{SMU id: pixel count} for the pixels touched by the boundary (EPSG:4326)."""
    s = status()
    if not s["raster_found"]:
        raise SoilDataUnavailable(
            f"HWSD raster not found at {s['raster']}. See backend/scripts/build_hwsd_sqlite.py for setup.")
    import rasterio
    from rasterio.mask import mask as rio_mask
    from shapely.geometry import shape

    with rasterio.open(raster_path()) as src:
        nodata = src.nodata
        counts: dict[int, int] = {}
        try:
            arr, _ = rio_mask(src, [boundary_geojson], crop=True, all_touched=True, filled=False)
            vals = np.ma.compressed(arr[0])
        except ValueError:      # boundary does not overlap the raster
            vals = np.array([], dtype=int)
        if vals.size == 0:      # tiny polygon that missed pixel centres: sample the centroid
            c = shape(boundary_geojson).centroid
            vals = np.array([v[0] for v in src.sample([(c.x, c.y)])])
        for v in vals:
            v = int(v)
            if v <= 0 or (nodata is not None and v == int(nodata)):
                continue
            counts[v] = counts.get(v, 0) + 1
    if not counts:
        raise SoilDataUnavailable("No HWSD soil data under this boundary (water body, urban area, or outside coverage).")
    return counts


# ---------------------------------------------------------------------------
# Attribute lookup + aggregation (pure: sqlite + numpy only)
# ---------------------------------------------------------------------------

def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if (np.isnan(f) or f < 0) else f


def load_layer_rows(smu_ids: list[int]) -> list[dict]:
    d = db_path()
    if not d.exists():
        raise SoilDataUnavailable(
            f"HWSD attribute database not found at {d}. Run backend/scripts/build_hwsd_sqlite.py.")
    conn = sqlite3.connect(d)
    conn.row_factory = sqlite3.Row
    try:
        q = ",".join("?" * len(smu_ids))
        rows = conn.execute(f"SELECT * FROM HWSD2_LAYERS WHERE HWSD2_SMU_ID IN ({q})", smu_ids).fetchall()
    finally:
        conn.close()
    return [{k.upper(): r[k] for k in r.keys()} for r in rows]


def aggregate_layers(rows: list[dict], smu_pixels: dict[int, int]) -> dict:
    """
    Weighted mean of each numeric property per depth layer.
    weight = (pixels of this SMU) x (soil unit SHARE %). Properties missing for some
    soil units are averaged over the units that DO have them (weights re-normalised).
    Also returns categorical distributions (drainage, soil group) using the same weights.
    """
    num = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))      # layer -> prop -> [sum w*v, sum w]
    drainage = defaultdict(float)
    groups = defaultdict(float)
    texture_codes = defaultdict(float)
    total_w = 0.0

    for r in rows:
        layer = str(r.get("LAYER", "")).upper()
        if layer not in LAYER_DEPTHS:
            continue
        smu = int(r["HWSD2_SMU_ID"])
        share = _num(r.get("SHARE"))
        w = smu_pixels.get(smu, 0) * (share if share is not None else 0.0)
        if w <= 0:
            continue
        for col, key in NUMERIC_PROPS.items():
            v = _num(r.get(col))
            if v is not None:
                acc = num[layer][key]
                acc[0] += w * v
                acc[1] += w
        if layer == "D1":       # categorical, counted once per soil unit (not per layer)
            total_w += w
            dr = _num(r.get("DRAINAGE"))
            if dr is not None and int(dr) in DRAINAGE_LABELS:
                drainage[int(dr)] += w
            name = r.get("WRB4") or r.get("FAO90") or r.get("WRB2")
            if name:
                groups[str(name).strip()] += w
            tx = r.get("TEXTURE_USDA")
            if tx not in (None, ""):
                texture_codes[str(tx)] += w

    layers = {L: {k: (acc[0] / acc[1]) for k, acc in props.items() if acc[1] > 0}
              for L, props in num.items()}
    if not layers.get("D1"):
        raise SoilDataUnavailable("HWSD has no usable topsoil attributes for the mapping unit(s) under this boundary.")

    def norm(d):
        t = sum(d.values())
        return {k: round(v / t, 3) for k, v in sorted(d.items(), key=lambda kv: -kv[1])} if t else {}

    return {"layers": layers, "drainage_distribution": norm(drainage),
            "soil_groups": norm(groups), "dominant_texture_code": (max(texture_codes, key=texture_codes.get)
                                                                   if texture_codes else None)}


def depth_average(layers: dict, top_cm: float, bot_cm: float) -> dict:
    """Thickness-weighted average of every property over [top_cm, bot_cm]."""
    acc = defaultdict(lambda: [0.0, 0.0])
    for L, (a, b) in LAYER_DEPTHS.items():
        thick = min(b, bot_cm) - max(a, top_cm)
        if thick <= 0 or L not in layers:
            continue
        for k, v in layers[L].items():
            acc[k][0] += thick * v
            acc[k][1] += thick
    return {k: s / t for k, (s, t) in acc.items() if t > 0}


def usda_texture(sand: float, silt: float, clay: float) -> str:
    """USDA texture class from sand/silt/clay percentages."""
    if None in (sand, silt, clay):
        return "unknown"
    if silt + 1.5 * clay < 15:
        return "sand"
    if silt + 1.5 * clay >= 15 and silt + 2 * clay < 30:
        return "loamy sand"
    if (7 <= clay < 20 and sand > 52 and silt + 2 * clay >= 30) or (clay < 7 and silt < 50 and silt + 2 * clay >= 30):
        return "sandy loam"
    if 7 <= clay < 27 and 28 <= silt < 50 and sand <= 52:
        return "loam"
    if (silt >= 50 and 12 <= clay < 27) or (50 <= silt < 80 and clay < 12):
        return "silt loam"
    if silt >= 80 and clay < 12:
        return "silt"
    if 20 <= clay < 35 and silt < 28 and sand > 45:
        return "sandy clay loam"
    if 27 <= clay < 40 and 20 < sand <= 45:
        return "clay loam"
    if 27 <= clay < 40 and sand <= 20:
        return "silty clay loam"
    if clay >= 35 and sand > 45:
        return "sandy clay"
    if clay >= 40 and silt >= 40:
        return "silty clay"
    if clay >= 40:
        return "clay"
    return "unclassified"


def _round_props(p: dict) -> dict:
    return {k: round(v, 2) for k, v in p.items()}


def profile_from_parts(agg: dict, smu_pixels: dict[int, int]) -> dict:
    """Shape the aggregate into the profile returned by the API."""
    layers = agg["layers"]
    top = depth_average(layers, 0, 30)
    sub = depth_average(layers, 30, 100)
    for p in (top, sub):
        if all(k in p for k in ("sand_pct", "silt_pct", "clay_pct")):
            p["texture_class"] = usda_texture(p["sand_pct"], p["silt_pct"], p["clay_pct"])
    notes = []
    n_px = sum(smu_pixels.values())
    if n_px <= 4:
        notes.append("Your boundary covers only a few ~1 km soil pixels, so these values describe the regional "
                     "soil association rather than your exact field.")
    if len(smu_pixels) > 1:
        notes.append(f"{len(smu_pixels)} soil mapping units overlap this boundary; values are pixel-weighted averages.")
    topout = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in top.items()}
    subout = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in sub.items()}
    return {
        "source": "HWSD v2.0 (FAO/IIASA)",
        "resolution": "~1 km (30 arc-seconds)",
        "pixels_sampled": n_px,
        "mapping_units": sorted(smu_pixels),
        "soil_groups": agg["soil_groups"],
        "drainage_distribution": {DRAINAGE_LABELS[k]: v for k, v in agg["drainage_distribution"].items()},
        "drainage_class_weights": agg["drainage_distribution"],
        "topsoil_0_30cm": topout,
        "subsoil_30_100cm": subout,
        "by_layer": {L: {"depth_cm": list(LAYER_DEPTHS[L]), **_round_props(v)} for L, v in sorted(layers.items())},
        "soil_test_needed": True,
        "notes": notes + ["Use for crop screening and planning. Fertilizer and lime rates need a laboratory soil test."],
    }


def soil_profile(boundary_geojson: dict) -> dict:
    smu = smu_histogram(boundary_geojson)
    rows = load_layer_rows(list(smu))
    if not rows:
        raise SoilDataUnavailable(f"Soil mapping unit(s) {sorted(smu)} not found in the attribute database "
                                  "(raster and database may be from different HWSD versions).")
    return profile_from_parts(aggregate_layers(rows, smu), smu)
