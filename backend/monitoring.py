"""
Field monitoring: turns an index raster into decisions an agronomist can act on.

  1. Observation history   - every processed image is saved (date, field stats,
                             data quality) so a field has a season timeline
                             instead of a single snapshot.
  2. Trend & alerts        - flags a field that is dropping vs its previous
                             observation or vs its own recent baseline.
  3. Problem areas         - finds contiguous patches that are clearly below
                             the field's norm, ranks them by impact, and gives
                             each a centroid so a scout knows where to walk.
  4. Plot performance      - ranks plots against the field baseline.

Design notes
  * Everything is relative to the field's own distribution (median / robust
    spread), so it works for any crop, stage or index without hand-tuned
    absolute thresholds.
  * These are *screening* tools. A low patch means "this differs from the rest
    of the field" - the cause (nutrients, water, pests, stand gaps, soil) must
    be confirmed on the ground. Responses say what to check first per index.
  * Natural phenology (senescence, harvest) lowers vegetation indices too, so
    trend alerts are prompts to look, not diagnoses.
"""
import os
import sqlite3
from datetime import date as _date
from pathlib import Path
from statistics import median

import numpy as np
from rasterio.crs import CRS
from rasterio.features import shapes as rio_shapes
from rasterio.transform import xy
from rasterio.warp import transform_geom
from scipy import ndimage as ndi
from shapely.geometry import shape, mapping

DB_PATH = Path(os.environ.get("FARMSCAN_DB", Path(__file__).parent / "farmscan.db"))

# Indices where a LOW value means a crop problem (the direction we screen for).
LOW_IS_BAD = {"ndvi", "ndre", "ndmi", "savi", "evi"}

# What to check first on the ground, per index.
CHECK_FIRST = {
    "ndvi": "Stand gaps / poor emergence, pests or disease, waterlogging, compaction.",
    "savi": "Stand gaps and thin canopy (soil-adjusted), poor emergence, early-season stress.",
    "evi": "Biomass shortfall in dense canopy: nutrient deficiency, disease, lodging.",
    "ndre": "Nitrogen / chlorophyll shortfall - take a leaf/tissue or SPAD reading.",
    "ndmi": "Crop water stress - check soil moisture, irrigation coverage, drainage, rooting.",
}

# Observations with less than this fraction of cloud-free field pixels are kept
# for the record but excluded from trend/alert logic (too unreliable).
MIN_VALID_FRACTION = 0.6

# Absolute change in the field mean that counts as a meaningful drop. Vegetation
# indices live on roughly a 0-1 scale, so 0.05 is a visible change in vigor.
DEFAULT_DROP_THRESHOLD = 0.05


# --------------------------------------------------------------------------
# Database
# --------------------------------------------------------------------------

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_monitoring_tables():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            boundary_id TEXT NOT NULL,
            index_name TEXT NOT NULL,
            obs_date TEXT NOT NULL,
            source TEXT NOT NULL,
            scene_id TEXT,
            cloud_cover REAL,
            valid_fraction REAL,
            mean REAL, median REAL, std REAL, p10 REAL, p90 REAL,
            pixel_count INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (boundary_id, index_name, obs_date, source)
        )
    """)
    conn.commit()
    conn.close()


def robust_stats(arr: np.ndarray) -> dict:
    """Mean/median/std/p10/p90 over valid (non-NaN) pixels."""
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return {"mean": None, "median": None, "std": None, "p10": None, "p90": None,
                "pixel_count": 0}
    return {
        "mean": float(np.mean(valid)), "median": float(np.median(valid)),
        "std": float(np.std(valid)),
        "p10": float(np.percentile(valid, 10)), "p90": float(np.percentile(valid, 90)),
        "pixel_count": int(valid.size),
    }


def save_observation(boundary_id: str, index_name: str, obs_date: str, source: str,
                     arr: np.ndarray, scene_id: str = None, cloud_cover: float = None,
                     valid_fraction: float = None) -> dict:
    """Persist one observation. Re-processing the same date/source replaces it."""
    st = robust_stats(arr)
    if valid_fraction is None:
        valid_fraction = 1.0
    conn = get_db()
    conn.execute(
        "INSERT OR REPLACE INTO observations (boundary_id, index_name, obs_date, source, "
        "scene_id, cloud_cover, valid_fraction, mean, median, std, p10, p90, pixel_count) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (boundary_id, index_name, obs_date, source, scene_id, cloud_cover, valid_fraction,
         st["mean"], st["median"], st["std"], st["p10"], st["p90"], st["pixel_count"]),
    )
    conn.commit()
    conn.close()
    return {"obs_date": obs_date, "source": source, "valid_fraction": valid_fraction, **st}


def list_observations(boundary_id: str, index_name: str, source: str = None) -> list[dict]:
    conn = get_db()
    q = "SELECT * FROM observations WHERE boundary_id = ? AND index_name = ?"
    params = [boundary_id, index_name]
    if source:
        q += " AND source = ?"
        params.append(source)
    q += " ORDER BY obs_date ASC"
    rows = [dict(r) for r in conn.execute(q, params).fetchall()]
    conn.close()
    return rows


# --------------------------------------------------------------------------
# Trend & alerts
# --------------------------------------------------------------------------

def analyze_trend(observations: list[dict], drop_threshold: float = DEFAULT_DROP_THRESHOLD) -> dict:
    """
    observations: rows from list_observations(), oldest first.

    Adds to each usable observation:
      change_vs_prev      mean minus the previous usable observation
      change_vs_baseline  mean minus the median of up to 4 earlier usable observations
    and raises an alert when either drop reaches `drop_threshold`.
    Observations below MIN_VALID_FRACTION are returned but marked usable=False.
    """
    series, alerts = [], []
    usable_means = []
    for o in observations:
        usable = (o["mean"] is not None and (o["valid_fraction"] or 0) >= MIN_VALID_FRACTION)
        item = {
            "obs_date": o["obs_date"], "source": o["source"], "mean": o["mean"],
            "p10": o["p10"], "p90": o["p90"], "valid_fraction": o["valid_fraction"],
            "cloud_cover": o["cloud_cover"], "usable": usable,
            "change_vs_prev": None, "change_vs_baseline": None,
        }
        if usable:
            if usable_means:
                item["change_vs_prev"] = o["mean"] - usable_means[-1]
                item["change_vs_baseline"] = o["mean"] - median(usable_means[-4:])
            usable_means.append(o["mean"])

            reasons = []
            if item["change_vs_prev"] is not None and item["change_vs_prev"] <= -drop_threshold:
                reasons.append(f"down {abs(item['change_vs_prev']):.2f} since the previous image")
            if (item["change_vs_baseline"] is not None
                    and item["change_vs_baseline"] <= -drop_threshold * 1.4):
                reasons.append(f"{abs(item['change_vs_baseline']):.2f} below the recent baseline")
            if reasons:
                item["alert"] = True
                alerts.append({"obs_date": o["obs_date"], "message": "; ".join(reasons)})
        series.append(item)

    usable_series = [s for s in series if s["usable"]]
    status = "insufficient_data"
    if len(usable_series) >= 2:
        d = usable_series[-1]["change_vs_prev"]
        status = "declining" if d <= -drop_threshold else "improving" if d >= drop_threshold else "stable"
    return {
        "series": series, "alerts": alerts, "status": status,
        "usable_count": len(usable_series),
        "excluded_count": len(series) - len(usable_series),
        "note": ("Alerts are prompts to inspect the field. Natural senescence or harvest also "
                 "lowers vegetation indices; confirm on the ground."),
    }


# --------------------------------------------------------------------------
# Raster helpers
# --------------------------------------------------------------------------

def _is_geographic(crs) -> bool:
    try:
        return CRS.from_user_input(crs).is_geographic
    except Exception:
        return False


def pixel_area_m2(transform, crs, shape_hw: tuple) -> float:
    """Ground area of one pixel in m^2 (handles metric CRS and degrees)."""
    if _is_geographic(crs):
        lat_mid = transform.f + transform.e * (shape_hw[0] / 2.0)
        m_lat = 111320.0
        m_lon = 111320.0 * np.cos(np.radians(lat_mid))
        return abs(transform.a) * m_lon * abs(transform.e) * m_lat
    return abs(transform.a * transform.e)


def _to_4326_geom(geom: dict, crs) -> dict:
    return geom if _is_geographic(crs) else transform_geom(crs, "EPSG:4326", geom)


def _to_4326_xy(x: float, y: float, crs) -> tuple:
    if _is_geographic(crs):
        return x, y
    g = transform_geom(crs, "EPSG:4326", {"type": "Point", "coordinates": (x, y)})
    return g["coordinates"][0], g["coordinates"][1]


# --------------------------------------------------------------------------
# Problem areas -> scouting points
# --------------------------------------------------------------------------

def find_problem_areas(arr: np.ndarray, transform, crs, index_name: str,
                       sensitivity: float = 1.5, min_area_m2: float = 400.0,
                       max_areas: int = 10) -> dict:
    """
    Finds contiguous patches that sit clearly below the field's norm.

    A pixel is "low" when it is more than `sensitivity` robust standard
    deviations below the field median (robust sigma = 1.4826 * MAD, so a few
    extreme pixels can't hide a problem). Speckle is removed, the remaining
    low pixels are grouped into patches, and patches smaller than
    `min_area_m2` are dropped. Patches are ranked by impact =
    area x how far below the field norm they are.

    Returns {"summary": {...}, "areas": [...]}, each area carrying a polygon and
    centroid in EPSG:4326 for the map and for scouting notes.
    """
    if index_name not in LOW_IS_BAD:
        raise ValueError(
            f"Problem-area screening needs a crop-vigor index where low values mean stress "
            f"({', '.join(sorted(LOW_IS_BAD))}). '{index_name}' is not one of them."
        )

    valid = ~np.isnan(arr)
    if valid.sum() < 25:
        raise ValueError("Not enough valid pixels to assess the field (heavy cloud or tiny boundary).")

    vals = arr[valid]
    med = float(np.median(vals))
    sigma = 1.4826 * float(np.median(np.abs(vals - med)))
    sigma = max(sigma, 0.02)          # floor: a perfectly uniform field must not flag noise
    threshold = med - sensitivity * sigma

    low = valid & (arr < threshold)
    low = ndi.binary_opening(low, structure=np.ones((3, 3), bool))
    labels, n = ndi.label(low, structure=np.ones((3, 3), bool))

    px_area = pixel_area_m2(transform, crs, arr.shape)
    field_area_m2 = float(valid.sum()) * px_area

    areas = []
    for lab in range(1, n + 1):
        mask = labels == lab
        area_m2 = float(mask.sum()) * px_area
        if area_m2 < min_area_m2:
            continue
        mean_val = float(arr[mask].mean())
        deficit = med - mean_val
        ys, xs = np.nonzero(mask)
        cx, cy = xy(transform, float(ys.mean()), float(xs.mean()))
        lon, lat = _to_4326_xy(cx, cy, crs)

        polys = [shape(g) for g, v in rio_shapes(mask.astype(np.uint8), mask=mask, transform=transform)]
        merged = polys[0] if len(polys) == 1 else max(polys, key=lambda p: p.area)
        geom4326 = _to_4326_geom(mapping(merged), crs)

        areas.append({
            "area_m2": round(area_m2, 1), "area_ha": round(area_m2 / 10000.0, 3),
            "pct_of_field": round(100.0 * area_m2 / field_area_m2, 1),
            "mean_value": round(mean_val, 3), "field_median": round(med, 3),
            "deficit": round(deficit, 3), "severity_sigma": round(deficit / sigma, 1),
            "impact": round(area_m2 * deficit, 1),
            "centroid": {"lat": lat, "lng": lon}, "geometry": geom4326,
        })

    areas.sort(key=lambda a: a["impact"], reverse=True)
    areas = areas[:max_areas]
    for rank, a in enumerate(areas, 1):
        a["rank"] = rank
        a["priority"] = "high" if a["severity_sigma"] >= 3 else "medium" if a["severity_sigma"] >= 2 else "watch"

    flagged_m2 = sum(a["area_m2"] for a in areas)
    return {
        "summary": {
            "index": index_name, "field_median": round(med, 3), "threshold": round(threshold, 3),
            "robust_sigma": round(sigma, 3), "sensitivity": sensitivity,
            "field_area_ha": round(field_area_m2 / 10000.0, 2),
            "flagged_area_ha": round(flagged_m2 / 10000.0, 3),
            "flagged_pct_of_field": round(100.0 * flagged_m2 / field_area_m2, 1),
            "area_count": len(areas),
            "check_first": CHECK_FIRST.get(index_name),
            "note": ("These patches differ from the rest of this field. Confirm the cause on the "
                     "ground before spending on inputs."),
        },
        "areas": areas,
    }


def scouting_note_text(area: dict, index_name: str, obs_date: str) -> str:
    return (f"[Scout #{area['rank']} · {area['priority']}] {index_name.upper()} "
            f"{area['mean_value']:.2f} vs field {area['field_median']:.2f} "
            f"({area['area_ha']:.2f} ha, {area['pct_of_field']:.0f}% of field, {obs_date}). "
            f"Check: {CHECK_FIRST.get(index_name, 'cause on the ground')}")


# --------------------------------------------------------------------------
# Plot performance
# --------------------------------------------------------------------------

def rank_plots(plot_stats: list[dict], watch_z: float = -1.0, under_z: float = -1.5) -> dict:
    """
    plot_stats: per-plot rows with 'mean' (None when no data), as returned by the
    zonal-stats logic. Ranks plots against the field baseline.

    z = (plot mean - median of plot means) / robust spread of plot means.
    Flags: 'underperforming' (z <= under_z), 'watch' (z <= watch_z), 'ok', 'top' (z >= 1).
    Also reports the uniformity of the field (CV of plot means) and the gap between the
    median plot and the top quartile - the headroom if weaker plots were brought up.
    """
    rows = [p for p in plot_stats if p.get("mean") is not None]
    if len(rows) < 3:
        raise ValueError("Need at least 3 plots with data to rank performance.")

    means = np.array([p["mean"] for p in rows], dtype=float)
    med = float(np.median(means))
    spread = 1.4826 * float(np.median(np.abs(means - med)))
    spread = max(spread, float(np.std(means)) * 0.5, 0.01)
    q75 = float(np.percentile(means, 75))

    out = []
    for p in rows:
        z = (p["mean"] - med) / spread
        flag = ("underperforming" if z <= under_z else "watch" if z <= watch_z
                else "top" if z >= 1.0 else "ok")
        out.append({**p, "z_score": round(z, 2), "vs_median": round(p["mean"] - med, 3),
                    "gap_to_top_quartile": round(max(0.0, q75 - p["mean"]), 3), "flag": flag})
    out.sort(key=lambda p: p["mean"])
    for rank, p in enumerate(out, 1):
        p["rank_from_bottom"] = rank

    cv = float(np.std(means) / abs(np.mean(means))) * 100 if np.mean(means) else None
    return {
        "plots": out,
        "summary": {
            "plots_ranked": len(out), "median_plot_mean": round(med, 3),
            "top_quartile_mean": round(q75, 3), "headroom_median_to_top_quartile": round(q75 - med, 3),
            "uniformity_cv_pct": round(cv, 1) if cv is not None else None,
            "underperforming": sum(1 for p in out if p["flag"] == "underperforming"),
            "watch": sum(1 for p in out if p["flag"] == "watch"),
            "unranked_no_data": len(plot_stats) - len(rows),
        },
    }
