"""
Weekly farm alerts: what needs a look on each farm this week, from data that comes in on its own.

  dry spell    in the rainy season, 10+ days in a row without useful rain (≥ 1 mm) up to today and little forecast
  heavy rain   a day of ≥ 50 mm forecast in the next week (waterlogging, erosion, washing off inputs)
  heat         a day of ≥ 38 °C forecast (heat stress, worst at flowering)
  fire         a satellite fire detection (NASA FIRMS, VIIRS 375 m on NOAA-20 and NOAA-21) on the farm or within 500 m in the last 7 days
  crop drop    the farm's NDVI or radar index fell in an image of the last 21 days (from its timeline)

Weather is modelled for the area (Open-Meteo, ~9–25 km), not measured on the farm; fire detections can be a cooking
fire or a burning field next door. Alerts are prompts to check, and say so.
"""
import csv
import io
import os
from datetime import date, timedelta

import numpy as np

DRY_SPELL_DAYS = 10
DRY_AHEAD_MM = 5.0
HEAVY_RAIN_MM = 50.0
HEAT_C = 38.0
FIRE_BUFFER_M = 500
# VIIRS 375 m near-real-time fire detections. Suomi NPP stopped delivering new data on 2 Nov 2026, so NOAA-20 and
# NOAA-21 are used (same instrument, same resolution, both flying).
FIRMS_SOURCES = ("VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT")
FIRE_DAYS = 7
CROP_DAYS = 21


def weather_alerts(wx: dict, wet_months: list[int] | None, today: date | None = None) -> list[dict]:
    """wx: {"dates", "rain", "tmax", "today_index"} (past days then forecast). wet_months: the area's rainy months
    (1-12) from its climate normals, or None if unknown (then dry spells are checked all year)."""
    today = today or date.today()
    rain = [0.0 if v is None else float(v) for v in wx.get("rain") or []]
    tmax = [None if v is None else float(v) for v in wx.get("tmax") or []]
    dates = wx.get("dates") or []
    t = int(wx.get("today_index", 0))
    out = []
    past, ahead = rain[:t + 1], rain[t + 1:]
    run = 0
    for r in reversed(past):
        if r >= 1.0:
            break
        run += 1
    in_season = wet_months is None or today.month in wet_months
    if in_season and run >= DRY_SPELL_DAYS and sum(ahead[:7]) < DRY_AHEAD_MM:
        out.append({"kind": "dry_spell", "level": "high" if run >= 18 else "medium",
                    "words": f"Dry spell: {run} days without useful rain in the rainy season, and under {DRY_AHEAD_MM:.0f} mm "
                             "forecast for the next week. Check soil moisture; hold top-dressing until rain."})
    heavy = [(dates[t + 1 + i], r) for i, r in enumerate(ahead) if r >= HEAVY_RAIN_MM and t + 1 + i < len(dates)]
    if heavy:
        d, r = max(heavy, key=lambda x: x[1])
        out.append({"kind": "heavy_rain", "level": "medium",
                    "words": f"Heavy rain forecast: {r:.0f} mm on {d}. Check drainage; avoid spraying or fertilising just before."})
    hot = [(dates[t + 1 + i], v) for i, v in enumerate(tmax[t + 1:]) if v is not None and v >= HEAT_C and t + 1 + i < len(dates)]
    if hot:
        d, v = max(hot, key=lambda x: x[1])
        out.append({"kind": "heat", "level": "medium",
                    "words": f"Heat: up to {v:.0f} °C forecast ({d}). Water stress risk, worst at flowering."})
    return out


def fire_alerts(detections: list[dict], geometry: dict) -> list[dict]:
    """Detections [{latitude, longitude, acq_date, confidence, frp}] on the farm or within FIRE_BUFFER_M of it."""
    from pyproj import Transformer
    from shapely.geometry import Point, shape
    from shapely.ops import transform

    farm = shape(geometry)
    c = farm.centroid
    zone = int((c.x + 180) // 6) + 1
    to_m = Transformer.from_crs("EPSG:4326", f"EPSG:{32600 + zone if c.y >= 0 else 32700 + zone}", always_xy=True).transform
    near = transform(to_m, farm).buffer(FIRE_BUFFER_M)
    on_farm = transform(to_m, farm)
    hits = []
    for d in detections:
        try:
            p = transform(to_m, Point(float(d["longitude"]), float(d["latitude"])))
        except (KeyError, ValueError):
            continue
        if near.contains(p):
            hits.append((d, on_farm.contains(p)))
    if not hits:
        return []
    inside = sum(1 for _, i in hits if i)
    last = max(str(d.get("acq_date", "")) for d, _ in hits)
    where = f"{inside} on the farm and {len(hits) - inside} within {FIRE_BUFFER_M} m" if inside else f"{len(hits)} within {FIRE_BUFFER_M} m of the farm"
    return [{"kind": "fire", "level": "high" if inside else "medium",
             "words": f"Fire: {len(hits)} satellite fire detection(s) in the last {FIRE_DAYS} days ({where}), latest {last}. "
                      "It may be a nearby field or bush burning: check the crop and stores."}]


def crop_alerts(trend: dict, index: str, today: date | None = None) -> list[dict]:
    """Drops in the farm's timeline (monitoring.analyze_trend) dated in the last CROP_DAYS days."""
    today = today or date.today()
    since = (today - timedelta(days=CROP_DAYS)).isoformat()
    recent = [a for a in trend.get("alerts", []) if a["obs_date"] >= since]
    if not recent:
        return []
    a = recent[-1]
    name = {"ndvi": "NDVI (crop vigour)", "rvi": "radar vegetation index"}.get(index, index.upper())
    return [{"kind": "crop_drop", "level": "high",
             "words": f"Crop: {name} {a['message']} (image of {a['obs_date']}). Find the problem areas and send a scout."}]


def fetch_fires(geometry: dict, days: int = FIRE_DAYS) -> list[dict] | None:
    """NASA FIRMS VIIRS detections around the farm (needs FIRMS_MAP_KEY; None when it is not set)."""
    key = os.environ.get("FIRMS_MAP_KEY")
    if not key:
        return None
    import requests
    from shapely.geometry import shape
    w, s, e, n = shape(geometry).buffer(0.01).bounds
    out, failed = [], 0
    for source in FIRMS_SOURCES:
        url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{source}/{w:.4f},{s:.4f},{e:.4f},{n:.4f}/{min(10, days)}"
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            if not r.text.lower().startswith("latitude"):            # FIRMS answers errors as plain text with 200
                raise ValueError(r.text[:200])
            out += list(csv.DictReader(io.StringIO(r.text)))
        except Exception:
            failed += 1
    if failed == len(FIRMS_SOURCES):
        raise RuntimeError("NASA FIRMS could not be reached (or refused the MAP_KEY)")
    return out


def email_text(org: str, farms: list[dict], link: str) -> tuple[str, str]:
    """(subject, body) of the weekly email. farms: [{"name", "alerts": [...]}] with at least one alert each."""
    n = sum(len(f["alerts"]) for f in farms)
    subject = f"FarmScan: {len(farms)} farm{'s' if len(farms) != 1 else ''} to check this week ({org})"
    lines = [f"This week {n} alert{'s' if n != 1 else ''} on {len(farms)} of {org}'s farms:", ""]
    for f in farms:
        lines.append(f"{f['name']}")
        lines += [f"  - {a['words']}" for a in f["alerts"]]
        lines.append("")
    lines += [f"Open FarmScan: {link}", "",
              "Weather is modelled for the area, not measured on the farm; fire detections may be nearby burning. "
              "These are prompts to check, not diagnoses.",
              "You get this because you asked for FarmScan's weekly alerts; switch them off in FarmScan."]
    return subject, "\n".join(lines)


def wet_months_of(climate: dict | None) -> list[int] | None:
    if not climate:
        return None
    m = climate.get("wet_months")
    return [int(x) for x in m] if m else None


def mean_or_none(v):
    v = [x for x in v if x is not None]
    return float(np.mean(v)) if v else None
