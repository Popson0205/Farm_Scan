"""
Crop suitability: combines HWSD soil constituents (soil.py) with weather history
(weather.py) into a ranked, explainable answer to "what can I grow here, when
should I plant, and what will limit me?".

Method (FAO-style land evaluation, simplified)
  1. Each factor (temperature, cycle rainfall, pH, organic carbon, clay, sand,
     CEC, salinity, coarse fragments, drainage) is scored 0-1 against the crop's
     requirement using a trapezoid: 0 outside the absolute limits, 1 inside the
     optimum, linear in between. Requirements live in crop_requirements.json.
  2. Climate factors (incl. rain in the first 30 days, which captures the onset of
     the rains) are scored per YEAR of the weather record and averaged, so a
     site with erratic rainfall scores lower than one with dependable rainfall.
  3. Class follows the Law of the Minimum (Liebig): the most limiting factor sets
     the class (S1 >= 0.8, S2 >= 0.6, S3 >= 0.4, else N). A 0-100 index (geometric
     mean of all factors) ranks crops within a class.
  4. The planting window is searched: every 7th start date of the year is scored
     on rainfall + temperature over the crop cycle; the best distinct windows
     are returned with dry-spell, failure-risk and heat statistics.

What this is NOT
  * It is a screening/planning tool built on ~1 km soil and ~10 km climate data.
    It cannot see your field's slope, depth to a hardpan, pests, or management.
    Confirm with a soil test and local extension advice before committing inputs.
"""
import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

import weather as weather_mod
from weather import WeatherIndex, window_across_years

CROPS_PATH = Path(os.environ.get("CROP_REQUIREMENTS", Path(__file__).parent / "crop_requirements.json"))

FACTOR_LABELS = {
    "tmean": ("Mean temperature over crop cycle", "°C"),
    "rain": ("Rainfall over crop cycle", "mm"),
    "establish": ("Rain in first 30 days (establishment)", "mm"),
    "ph": ("Soil pH (water)", ""),
    "organic_carbon": ("Organic carbon (topsoil)", "%"),
    "clay": ("Clay (topsoil)", "%"),
    "sand": ("Sand (topsoil)", "%"),
    "cec": ("Cation exchange capacity (topsoil)", "cmol/kg"),
    "ec": ("Salinity (EC)", "dS/m"),
    "coarse": ("Coarse fragments", "vol %"),
    "drainage": ("Natural drainage", ""),
}
# soil factor -> key in the HWSD topsoil profile
SOIL_KEYS = {"ph": "ph_water", "organic_carbon": "organic_carbon_pct", "clay": "clay_pct",
             "sand": "sand_pct", "cec": "cec_cmol_kg", "ec": "ec_ds_m", "coarse": "coarse_frag_pct"}

CLASSES = [(0.8, "S1", "Highly suitable"), (0.6, "S2", "Moderately suitable"),
           (0.4, "S3", "Marginally suitable"), (0.0, "N", "Not suitable")]
CLASS_ORDER = {"S1": 0, "S2": 1, "S3": 2, "N": 3}
MIN_FACTORS_REQUIRED = 5
MIN_YEARS_REQUIRED = 5

HINTS = {
    ("ph", "low"): "Soil is too acidic for this crop. Get a lime-requirement test; liming and organic matter raise pH and ease aluminium/manganese toxicity.",
    ("ph", "high"): "Soil is alkaline for this crop; phosphorus and micronutrient lock-up is likely. Favour organic amendments and confirm with a soil test.",
    ("organic_carbon", "low"): "Low organic carbon: build it with crop residues, mulch, compost/manure and legume rotations; split fertilizer applications.",
    ("clay", "low"): "Light, free-draining soil holds little water and nutrients: use mulch and organic matter, split fertilizer, and expect drought stress in dry spells.",
    ("sand", "high"): "Sandy soil holds little water and nutrients: use mulch and organic matter, split fertilizer, and expect drought stress in dry spells.",
    ("clay", "high"): "Heavy soil: waterlogging and poor workability. Use ridges/raised beds and avoid cultivating when wet.",
    ("cec", "low"): "Low nutrient-holding capacity: apply fertilizer in small split doses and raise organic matter to reduce leaching losses.",
    ("ec", "high"): "Salinity risk: ensure drainage, avoid saline irrigation water, and consider salt-tolerant crops.",
    ("coarse", "high"): "Stony/gravelly soil limits rooting depth, water storage and tillage.",
    ("rain", "low"): "Cycle rainfall is below what this crop needs in a typical year: plan supplementary irrigation, water harvesting/mulching, an earlier-maturing variety, or a more drought-tolerant crop.",
    ("rain", "high"): "Rainfall is above this crop's optimum: prioritise drainage, ridges, and disease/erosion management.",
    ("establish", "low"): "Too little rain in the first month after planting: seedlings may fail. Wait for the rains to establish (e.g. a few good rainfalls without a long dry gap), or irrigate/mulch if planting early.",
    ("tmean", "low"): "Cool conditions will slow growth and lengthen the cycle; check altitude-adapted varieties.",
    ("tmean", "high"): "Hot conditions raise heat and water stress: choose heat-tolerant varieties and time flowering away from the hottest weeks.",
    ("drainage", "wet"): "Soils here tend to waterlog: use ridges/raised beds and drainage channels, or consider a waterlogging-tolerant crop such as rice.",
    ("drainage", "dry"): "Soils here drain very fast: mulch and organic matter help hold water; irrigation may be needed.",
}


# ---------------------------------------------------------------------------
# Crop table + scoring primitives
# ---------------------------------------------------------------------------

def load_crops() -> dict:
    with open(CROPS_PATH, encoding="utf-8") as f:
        return json.load(f)["crops"]


def trap(x, a, b, c, d):
    """Trapezoid membership. None for c/d means open-ended (no upper optimum / limit)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    c = np.inf if c is None else c
    d = np.inf if d is None else d
    if x < a or x > d:
        return 0.0
    if b <= x <= c:
        return 1.0
    if x < b:
        return (x - a) / (b - a) if b > a else 1.0
    return (d - x) / (d - c) if d > c else 1.0


def rating(score: float) -> str:
    return "good" if score >= 0.8 else "moderate" if score >= 0.6 else "poor" if score >= 0.4 else "very poor"


def classify(min_score: float) -> tuple[str, str]:
    for lo, code, label in CLASSES:
        if min_score >= lo:
            return code, label
    return "N", "Not suitable"


def drainage_score(weights: dict, table: dict) -> float | None:
    """Share-weighted drainage score. `weights` = {class number: share}."""
    if not weights:
        return None
    tot = sum(weights.values())
    return sum(w * table.get(str(int(k)), 0.0) for k, w in weights.items()) / tot if tot else None


def _hint_direction(key: str, value: float, req: list) -> str:
    a, b, c, d = req
    return "low" if value < b else "high"


# ---------------------------------------------------------------------------
# Climate for a crop: scored per year, averaged
# ---------------------------------------------------------------------------

def season_stats(idx: WeatherIndex, crop: dict, month: int, day: int, extras: bool = True) -> dict | None:
    """Per-year rain/temperature for a crop cycle starting on (month, day), with risk numbers.
    extras=False skips the per-day loops (dry spells, heat days) -- used while scanning dates."""
    cycle = crop["cycle_days"]
    yrs = window_across_years(idx, month, day, cycle)
    if len(yrs) < MIN_YEARS_REQUIRED:
        return None
    rq = crop["factors"]
    rain_scores = [trap(y["rain_mm"], *rq["rain"]) for y in yrs]
    # establishment: rain in the first 30 days. Without it, a window can look fine on cycle totals
    # while the seedlings die in a pre-onset dry spell.
    est_vals = []
    for y in yrs:
        w30 = idx.window(date(y["year"], month, day), 30)
        est_vals.append(w30["rain_mm"] if w30 else np.nan)
    est_vals = np.array(est_vals)
    est_scores = [trap(float(v), *rq["establish"]) if not np.isnan(v) else None for v in est_vals]
    if any(s_ is None for s_ in est_scores):
        return None
    temp_scores = [trap(y["tmean_c"], *rq["tmean"]) for y in yrs]
    rains = np.array([y["rain_mm"] for y in yrs])
    dry, heat = [], []
    if extras:
        ex = [idx.window_extras(date(y["year"], month, day), cycle) for y in yrs]
        dry = [e["max_dry_spell_days"] for e in ex]
        heat = [e["heat_days"] for e in ex if e["heat_days"] is not None]
    return {
        "n_years": len(yrs),
        "rain_score": float(np.mean(rain_scores)),
        "temp_score": float(np.mean(temp_scores)),
        "establish_score": float(np.mean(est_scores)),
        "establish_rain_mm": float(est_vals.mean()),
        "combined": float(np.mean([r * t * e for r, t, e in zip(rain_scores, temp_scores, est_scores)])),
        "rain_mean_mm": float(rains.mean()),
        "rain_p20_mm": float(np.percentile(rains, 20)),
        "tmean_c": float(np.mean([y["tmean_c"] for y in yrs])),
        "years_below_min_rain": int(np.sum(rains < rq["rain"][1])),   # below the OPTIMUM minimum
        "years_below_abs_min_rain": int(np.sum(rains < rq["rain"][0])),
        "longest_dry_spell_days_avg": round(float(np.mean(dry)), 1) if dry else None,
        "longest_dry_spell_days_worst": int(max(dry)) if dry else None,
        "heat_days_avg": round(float(np.mean(heat)), 1) if heat else None,
    }


def _label_date(month: int, day: int) -> str:
    d = date(2023, month, day)          # not strftime("%-d"): that fails on Windows
    return f"{d.day} {d:%b}"


def planting_windows(idx: WeatherIndex, crop: dict, top: int = 3, step_days: int = 7) -> list[dict]:
    """Best distinct (>= 45 days apart) start dates for a crop, by climate score."""
    cycle = crop["cycle_days"]
    cands = []
    for k in range(0, 365, step_days):
        d0 = date(2023, 1, 1) + timedelta(days=k)
        s = season_stats(idx, crop, d0.month, d0.day, extras=False)
        if s:
            cands.append((s["combined"], d0, s))
    cands.sort(key=lambda t: -t[0])
    chosen = []
    for score, d0, s in cands:
        if all(min((d0 - c[1]).days % 365, (c[1] - d0).days % 365) >= 45 for c in chosen):
            chosen.append((score, d0, s))
        if len(chosen) >= top:
            break
    out = []
    for score, d0, s in chosen:
        s = season_stats(idx, crop, d0.month, d0.day)      # full stats only for the winners
        end = d0 + timedelta(days=cycle)
        out.append({
            "plant_from": _label_date(d0.month, d0.day), "harvest_about": _label_date(end.month, end.day),
            "start_month": d0.month, "start_day": d0.day,
            "climate_score": round(score, 2),
            "rain_mean_mm": round(s["rain_mean_mm"]), "rain_dry_year_mm": round(s["rain_p20_mm"]),
            "first_30_days_rain_mm": round(s["establish_rain_mm"]),
            "years_below_optimum_rain": f'{s["years_below_min_rain"]} of {s["n_years"]}',
            "longest_dry_spell_days_avg": s["longest_dry_spell_days_avg"],
            "longest_dry_spell_days_worst": s["longest_dry_spell_days_worst"],
            "heat_days_avg": s["heat_days_avg"],
        })
    return out


# ---------------------------------------------------------------------------
# Scoring one crop
# ---------------------------------------------------------------------------

def score_crop(crop_id: str, crop: dict, profile: dict, idx: WeatherIndex,
               start: tuple[int, int] | None = None) -> dict:
    top = profile["topsoil_0_30cm"]
    rq = crop["factors"]
    factors, missing = [], []

    # --- climate (per-year scored) at the chosen or best planting date
    windows = []
    if start is None:
        windows = planting_windows(idx, crop)
        best = windows[0] if windows else None
        start = (best["start_month"], best["start_day"]) if best else None
    season = season_stats(idx, crop, *start) if start else None

    if season:
        factors.append(_factor("tmean", season["tmean_c"], season["temp_score"], rq["tmean"]))
        factors.append(_factor("rain", season["rain_mean_mm"], season["rain_score"], rq["rain"]))
        factors.append(_factor("establish", season["establish_rain_mm"], season["establish_score"], rq["establish"]))
    else:
        missing += ["tmean", "rain", "establish"]

    # --- soil
    for key, hwsd_key in SOIL_KEYS.items():
        v = top.get(hwsd_key)
        s = trap(v, *rq[key]) if v is not None else None
        if s is None:
            missing.append(key)
        else:
            factors.append(_factor(key, v, s, rq[key]))

    ds = drainage_score(profile.get("drainage_class_weights") or {}, crop["drainage"])
    if ds is None:
        missing.append("drainage")
    else:
        factors.append({"key": "drainage", "label": FACTOR_LABELS["drainage"][0], "unit": "",
                        "value": _dominant_drainage(profile), "score": round(ds, 2), "rating": rating(ds)})

    if len(factors) < MIN_FACTORS_REQUIRED:
        return {"crop": crop_id, "label": crop["label"], "error":
                f"Only {len(factors)} usable factors (need {MIN_FACTORS_REQUIRED}). Missing: {', '.join(missing)}"}

    scores = np.array([f["score"] for f in factors])
    index = float(np.exp(np.mean(np.log(np.maximum(scores, 0.01)))) * 100)
    worst = min(factors, key=lambda f: f["score"])
    code, label = classify(worst["score"])

    recs = []
    for f in sorted(factors, key=lambda f: f["score"]):
        if f["score"] >= 0.8:
            break
        h = _recommendation(f, rq, profile)
        if h:
            recs.append({"factor": f["key"], "severity": f["rating"], "advice": h})

    return {
        "crop": crop_id, "label": crop["label"], "cycle_days": crop["cycle_days"],
        "class": code, "class_label": label, "index": round(index),
        "limiting_factor": {"key": worst["key"], "label": worst["label"], "score": worst["score"],
                            "significant": worst["score"] < 0.8},     # False = nothing really limits this crop
        "factors": factors, "missing_factors": missing,
        "planting": {"chosen_start": _label_date(*start) if start else None,
                     "alternatives": windows,
                     "season": ({k: (round(v, 1) if isinstance(v, float) else v) for k, v in season.items()}
                                if season else None)},
        "recommendations": recs,
    }


def _factor(key, value, score, req):
    label, unit = FACTOR_LABELS[key]
    return {"key": key, "label": label, "unit": unit, "value": round(float(value), 2),
            "score": round(float(score), 2), "rating": rating(score), "optimum": [req[1], req[2]]}


def _dominant_drainage(profile) -> str:
    d = profile.get("drainage_distribution") or {}
    return next(iter(d), "unknown")


def _recommendation(f: dict, rq: dict, profile: dict) -> str | None:
    key = f["key"]
    if key == "drainage":
        w = profile.get("drainage_class_weights") or {}
        wet = sum(v for k, v in w.items() if int(k) >= 5)
        dry = sum(v for k, v in w.items() if int(k) <= 2)
        return HINTS.get(("drainage", "wet" if wet >= dry else "dry"))
    return HINTS.get((key, _hint_direction(key, f["value"], rq[key])))


# ---------------------------------------------------------------------------
# Crop-independent soil reading
# ---------------------------------------------------------------------------

def soil_interpretation(profile: dict) -> list[dict]:
    """Plain-language reading of the topsoil (what the numbers mean), not tied to a crop."""
    t = profile["topsoil_0_30cm"]
    out = []

    def add(name, value, unit, level, msg):
        out.append({"property": name, "value": round(value, 2), "unit": unit, "level": level, "meaning": msg})

    if (v := t.get("ph_water")) is not None:
        lvl = ("strongly acidic" if v < 5.0 else "acidic" if v < 5.5 else "slightly acidic" if v < 6.5
               else "neutral" if v <= 7.5 else "alkaline")
        add("pH (water)", v, "", lvl, "Most crops do best around pH 5.5-7.5; below ~5.5 liming is usually considered.")
    if (v := t.get("organic_carbon_pct")) is not None:
        lvl = "low" if v < 1.0 else "moderate" if v < 2.0 else "good"
        add("Organic carbon", v, "%", lvl, "Drives water holding, nutrient supply and soil structure.")
    if (v := t.get("cec_cmol_kg")) is not None:
        lvl = "low" if v < 6 else "moderate" if v < 12 else "good"
        add("CEC", v, "cmol/kg", lvl, "Ability to hold nutrients; low CEC soils leach fertilizer quickly.")
    if (v := t.get("base_sat_pct")) is not None:
        add("Base saturation", v, "%", "low" if v < 35 else "moderate" if v < 70 else "high",
            "Share of the exchange complex held by Ca, Mg, K, Na; low values go with acid, leached soils.")
    if (v := t.get("ec_ds_m")) is not None:
        add("Salinity (EC)", v, "dS/m", "saline" if v >= 4 else "slightly saline" if v >= 2 else "non-saline",
            "Salts reduce water uptake; above ~4 dS/m many crops are affected.")
    if (v := t.get("exch_sodium_pct")) is not None:
        add("Exchangeable sodium", v, "%", "sodic" if v >= 15 else "normal",
            "Sodic soils disperse, seal and drain poorly.")
    if t.get("texture_class"):
        out.append({"property": "Texture", "value": t["texture_class"], "unit": "", "level": t["texture_class"],
                    "meaning": f'{t.get("sand_pct", "?")}% sand, {t.get("silt_pct", "?")}% silt, '
                               f'{t.get("clay_pct", "?")}% clay.'})
    return out


# ---------------------------------------------------------------------------
# Top-level analysis
# ---------------------------------------------------------------------------

def parse_planting_date(s: str | None) -> tuple[int, int] | None:
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%m-%d"):
        try:
            d = datetime.strptime(s, fmt)
            return d.month, d.day
        except ValueError:
            continue
    raise ValueError("planting_date must be YYYY-MM-DD or MM-DD")


def analyze(profile: dict, series: dict, crop_ids: list[str] | None = None,
            planting_date: str | None = None, lat: float | None = None, lon: float | None = None) -> dict:
    crops = load_crops()
    ids = crop_ids or list(crops)
    unknown = [c for c in ids if c not in crops]
    if unknown:
        raise ValueError(f"Unknown crop(s): {', '.join(unknown)}. Available: {', '.join(crops)}")
    start = parse_planting_date(planting_date)
    idx = WeatherIndex(series)

    results = [score_crop(c, crops[c], profile, idx, start) for c in ids]
    ok = [r for r in results if "error" not in r]
    ok.sort(key=lambda r: (CLASS_ORDER[r["class"]], -r["index"]))
    failed = [r for r in results if "error" in r]

    caveats = list(profile.get("notes", []))
    caveats.append("Climate comes from gridded reanalysis/satellite-model data, not a gauge at your field. "
                   "Scores assume rainfed production unless you irrigate.")
    if series.get("stale"):
        caveats.append("Weather data came from an older cached copy because the provider was unreachable.")

    return {
        "location": {"lat": lat, "lon": lon},
        "method": "FAO-style land evaluation: limiting-factor class + geometric-mean index. "
                  "Requirements: crop_requirements.json (editable).",
        "soil": {"groups": profile.get("soil_groups"), "drainage": profile.get("drainage_distribution"),
                 "topsoil_0_30cm": profile["topsoil_0_30cm"], "interpretation": soil_interpretation(profile)},
        "climate": weather_mod.climate_summary(series),
        "ranking": ok, "not_scored": failed,
        "planting_date_mode": "user-specified" if start else "best window searched per crop",
        "caveats": caveats,
    }
