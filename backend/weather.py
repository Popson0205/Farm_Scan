"""
Open weather data for site suitability and planning. No API keys.

Sources
  * Open-Meteo (primary)  - ERA5 reanalysis archive (history back to 1940) and a
    forecast API. https://open-meteo.com  (free tier is for NON-COMMERCIAL use;
    a commercial FarmScan needs an Open-Meteo API key/subscription -- set
    OPEN_METEO_API_KEY and OPEN_METEO_BASE_URL if you have one.)
  * NASA POWER (fallback) - daily agroclimatology, free for any use.
    https://power.larc.nasa.gov

What this module gives the rest of the app
  * get_daily_history()   -> a contiguous daily series (tmax/tmin/tmean/rain/et0)
  * get_forecast()        -> next N days + simple, rule-based field advisories
  * monthly_climatology() -> long-term monthly rainfall / temperature table
  * WeatherIndex          -> O(1) rainfall / temperature sums over any date
                             window, used to find planting windows per crop

Accuracy notes
  * Reanalysis grids are ~9-25 km. They describe the *area*, not your field:
    rainfall in particular is patchy in the tropics. Treat results as
    planning-grade, and prefer a local rain gauge when you have one.
  * Responses are cached in SQLite (history: 30 days, forecast: 3 hours) so the
    app stays fast, polite to the free APIs, and usable through short outages.
"""
import json
import os
import sqlite3
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import requests

DB_PATH = Path(os.environ.get("FARMSCAN_DB", Path(__file__).parent / "farmscan.db"))

OPEN_METEO_BASE = os.environ.get("OPEN_METEO_BASE_URL", "")  # optional commercial host
OPEN_METEO_KEY = os.environ.get("OPEN_METEO_API_KEY", "")
_ARCHIVE_URL = (OPEN_METEO_BASE.rstrip("/") + "/v1/archive") if OPEN_METEO_BASE \
    else "https://archive-api.open-meteo.com/v1/archive"
_FORECAST_URL = (OPEN_METEO_BASE.rstrip("/") + "/v1/forecast") if OPEN_METEO_BASE \
    else "https://api.open-meteo.com/v1/forecast"
NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

DAILY_VARS = ("temperature_2m_max,temperature_2m_min,temperature_2m_mean,"
              "precipitation_sum,et0_fao_evapotranspiration")

HISTORY_TTL_S = 30 * 24 * 3600
FORECAST_TTL_S = 3 * 3600
HTTP_TIMEOUT_S = 40

RAINY_DAY_MM = 1.0           # a day counts as "wet" at/above this
HEAVY_RAIN_MM = 50.0         # advisory threshold (flooding / erosion / disease)
HEAT_TMAX_C = 35.0           # heat-stress threshold for most field crops


class WeatherUnavailable(RuntimeError):
    """Raised when no provider could supply data (and nothing usable is cached)."""


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def _db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS api_cache ("
                 "key TEXT PRIMARY KEY, payload TEXT NOT NULL, fetched_at REAL NOT NULL)")
    return conn


def _cache_get(key: str, ttl_s: float, allow_stale: bool = False):
    conn = _db()
    try:
        row = conn.execute("SELECT payload, fetched_at FROM api_cache WHERE key = ?", (key,)).fetchone()
    finally:
        conn.close()
    if not row:
        return None
    age = time.time() - row[1]
    if age <= ttl_s or allow_stale:
        return {"payload": json.loads(row[0]), "age_s": age, "stale": age > ttl_s}
    return None


def _cache_put(key: str, payload) -> None:
    conn = _db()
    try:
        conn.execute("INSERT OR REPLACE INTO api_cache (key, payload, fetched_at) VALUES (?, ?, ?)",
                     (key, json.dumps(payload), time.time()))
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Provider calls (network edge -- everything below this is pure and testable)
# ---------------------------------------------------------------------------

def _get_json(url: str, params: dict) -> dict:
    is_open_meteo = "open-meteo" in url or bool(OPEN_METEO_BASE and url.startswith(OPEN_METEO_BASE))
    if OPEN_METEO_KEY and is_open_meteo:
        params = {**params, "apikey": OPEN_METEO_KEY}
    r = requests.get(url, params=params, timeout=HTTP_TIMEOUT_S)
    if r.status_code != 200:
        raise WeatherUnavailable(f"{url} -> HTTP {r.status_code}: {r.text[:200]}")
    return r.json()


def _fetch_open_meteo_history(lat: float, lon: float, start: date, end: date) -> dict:
    j = _get_json(_ARCHIVE_URL, {
        "latitude": round(lat, 4), "longitude": round(lon, 4),
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        "daily": DAILY_VARS, "timezone": "UTC",
    })
    d = j.get("daily") or {}
    if not d.get("time"):
        raise WeatherUnavailable("Open-Meteo archive returned no daily data")
    return {
        "dates": d["time"],
        "tmax": d.get("temperature_2m_max"), "tmin": d.get("temperature_2m_min"),
        "tmean": d.get("temperature_2m_mean"), "rain": d.get("precipitation_sum"),
        "et0": d.get("et0_fao_evapotranspiration"),
        "source": "Open-Meteo (ERA5 reanalysis)",
    }


def _fetch_nasa_power_history(lat: float, lon: float, start: date, end: date) -> dict:
    j = _get_json(NASA_POWER_URL, {
        "parameters": "T2M_MAX,T2M_MIN,T2M,PRECTOTCORR", "community": "AG",
        "latitude": round(lat, 4), "longitude": round(lon, 4),
        "start": start.strftime("%Y%m%d"), "end": end.strftime("%Y%m%d"), "format": "JSON",
    })
    p = (j.get("properties") or {}).get("parameter") or {}
    if not p.get("PRECTOTCORR"):
        raise WeatherUnavailable("NASA POWER returned no daily data")
    keys = sorted(p["PRECTOTCORR"].keys())

    def col(name):
        # POWER marks missing values as -999
        return [None if (p[name].get(k) is None or p[name][k] <= -900) else p[name][k] for k in keys]

    return {
        "dates": [f"{k[:4]}-{k[4:6]}-{k[6:]}" for k in keys],
        "tmax": col("T2M_MAX"), "tmin": col("T2M_MIN"), "tmean": col("T2M"),
        "rain": col("PRECTOTCORR"), "et0": [None] * len(keys),
        "source": "NASA POWER (daily agroclimatology)",
    }


def _fetch_open_meteo_forecast(lat: float, lon: float, days: int) -> dict:
    j = _get_json(_FORECAST_URL, {
        "latitude": round(lat, 4), "longitude": round(lon, 4),
        "daily": DAILY_VARS + ",precipitation_probability_max",
        "forecast_days": days, "timezone": "auto",
    })
    d = j.get("daily") or {}
    if not d.get("time"):
        raise WeatherUnavailable("Open-Meteo forecast returned no daily data")
    return {
        "dates": d["time"],
        "tmax": d.get("temperature_2m_max"), "tmin": d.get("temperature_2m_min"),
        "rain": d.get("precipitation_sum"), "rain_prob": d.get("precipitation_probability_max"),
        "et0": d.get("et0_fao_evapotranspiration"),
        "source": "Open-Meteo forecast",
    }


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

def get_daily_history(lat: float, lon: float, years: int = 10, end: date | None = None) -> dict:
    """
    Contiguous daily series for the last `years` COMPLETE calendar years.
    Returns {"dates", "tmax", "tmin", "tmean", "rain", "et0", "source", "cached", "stale"}
    with None for missing values.
    """
    years = max(3, min(int(years), 30))
    end = end or date(date.today().year - 1, 12, 31)
    start = date(end.year - years + 1, 1, 1)
    key = f"hist:{round(lat, 2)}:{round(lon, 2)}:{start}:{end}"

    hit = _cache_get(key, HISTORY_TTL_S)
    if hit:
        return {**hit["payload"], "cached": True, "stale": False}

    errors = []
    for fetch in (_fetch_open_meteo_history, _fetch_nasa_power_history):
        try:
            series = fetch(lat, lon, start, end)
            _cache_put(key, series)
            return {**series, "cached": False, "stale": False}
        except (WeatherUnavailable, requests.RequestException, ValueError) as e:
            errors.append(f"{fetch.__name__}: {e}")

    stale = _cache_get(key, HISTORY_TTL_S, allow_stale=True)
    if stale:   # an old copy of 10-year climate is still far better than nothing
        return {**stale["payload"], "cached": True, "stale": True}
    raise WeatherUnavailable("No weather provider reachable. " + " | ".join(errors))


class WeatherIndex:
    """
    Fast window statistics over a contiguous daily series (prefix sums), so the
    planting-window search can test hundreds of start dates x years cheaply.
    Missing days are tolerated: a window is only usable if >= min_valid of its
    days have data.
    """

    def __init__(self, series: dict):
        d = np.array(series["dates"], dtype="datetime64[D]")
        order = np.argsort(d)
        d = d[order]
        self.first = d[0]
        n = int((d[-1] - d[0]).astype(int)) + 1
        pos = (d - self.first).astype(int)

        def col(name):
            raw = series.get(name)
            out = np.full(n, np.nan)
            if raw is not None:
                arr = np.array([np.nan if v is None else v for v in raw], dtype=float)[order]
                out[pos] = arr
            return out

        self.n = n
        self.rain = col("rain")
        self.tmax = col("tmax")
        self.tmin = col("tmin")
        tmean = col("tmean")
        # derive tmean from tmax/tmin when the provider did not give it
        fill = np.isnan(tmean) & ~np.isnan(self.tmax) & ~np.isnan(self.tmin)
        tmean[fill] = (self.tmax[fill] + self.tmin[fill]) / 2
        self.tmean = tmean
        self.et0 = col("et0")

        def cs(a):
            return np.concatenate([[0.0], np.cumsum(np.nan_to_num(a))])

        def cc(a):
            return np.concatenate([[0], np.cumsum(~np.isnan(a))])

        self._rain_s, self._rain_c = cs(self.rain), cc(self.rain)
        self._tmean_s, self._tmean_c = cs(self.tmean), cc(self.tmean)
        self.first_year = int(str(self.first)[:4])
        self.last_date = d[-1]

    def _i(self, day: date) -> int:
        return int((np.datetime64(day) - self.first).astype(int))

    def window(self, start: date, length: int, min_valid: float = 0.9) -> dict | None:
        """Rain total (mm) and mean temp (C) for [start, start+length). None if data too thin."""
        i = self._i(start)
        j = i + length
        if i < 0 or j > self.n:
            return None
        rc = self._rain_c[j] - self._rain_c[i]
        tc = self._tmean_c[j] - self._tmean_c[i]
        if rc < min_valid * length or tc < min_valid * length:
            return None
        # scale rain up if a few days were missing, so totals stay comparable
        rain = (self._rain_s[j] - self._rain_s[i]) * (length / rc)
        tmean = (self._tmean_s[j] - self._tmean_s[i]) / tc
        return {"rain_mm": float(rain), "tmean_c": float(tmean)}

    def window_extras(self, start: date, length: int) -> dict:
        """Heavier stats (loops) -- call only for the window you actually report."""
        i, j = self._i(start), self._i(start) + length
        r = self.rain[i:j]
        wet = r >= RAINY_DAY_MM
        longest = run = 0
        for is_wet, valid in zip(wet, ~np.isnan(r)):
            if valid and not is_wet:
                run += 1
                longest = max(longest, run)
            else:
                run = 0
        tmx = self.tmax[i:j]
        return {
            "max_dry_spell_days": int(longest),
            "heat_days": int(np.nansum(tmx >= HEAT_TMAX_C)) if np.any(~np.isnan(tmx)) else None,
            "heavy_rain_days": int(np.nansum(r >= HEAVY_RAIN_MM)),
            "et0_mm": float(np.nansum(self.et0[i:j])) if np.any(~np.isnan(self.et0[i:j])) else None,
        }

    def years_covered(self) -> list[int]:
        return list(range(self.first_year, int(str(self.last_date)[:4]) + 1))


def window_across_years(idx: WeatherIndex, month: int, day: int, length: int) -> list[dict]:
    """The same calendar window (e.g. start 15 May, 120 days) in every year of the record."""
    out = []
    for y in idx.years_covered():
        try:
            w = idx.window(date(y, month, day), length)
        except ValueError:      # 29 Feb in a non-leap year
            continue
        if w:
            out.append({"year": y, **w})
    return out


# ---------------------------------------------------------------------------
# Climatology
# ---------------------------------------------------------------------------

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def monthly_climatology(series: dict) -> list[dict]:
    """Long-term average per calendar month (rain is a monthly TOTAL averaged over years)."""
    idx = WeatherIndex(series)
    dates = idx.first + np.arange(idx.n)
    months = (dates.astype("datetime64[M]").astype(int) % 12) + 1
    years = dates.astype("datetime64[Y]").astype(int) + 1970
    rows = []
    for m in range(1, 13):
        sel = months == m
        per_year_rain = []
        for y in np.unique(years[sel]):
            r = idx.rain[sel & (years == y)]
            if np.sum(~np.isnan(r)) >= 0.8 * len(r):
                per_year_rain.append(float(np.nansum(r)))
        rows.append({
            "month": MONTHS[m - 1],
            "rain_mm": round(float(np.mean(per_year_rain)), 1) if per_year_rain else None,
            "rain_mm_p20": round(float(np.percentile(per_year_rain, 20)), 1) if per_year_rain else None,
            "rainy_days": round(float(np.nansum(idx.rain[sel] >= RAINY_DAY_MM) / max(1, len(np.unique(years[sel])))), 1),
            "tmean_c": _r(np.nanmean(idx.tmean[sel])),
            "tmax_c": _r(np.nanmean(idx.tmax[sel])),
            "tmin_c": _r(np.nanmean(idx.tmin[sel])),
            "et0_mm": (round(float(np.nansum(idx.et0[sel]) / max(1, len(np.unique(years[sel])))), 1)
                       if np.any(~np.isnan(idx.et0[sel])) else None),
        })
    return rows


def climate_summary(series: dict) -> dict:
    """Headline numbers + the rainy season (months averaging >= 60 mm and >= ET0/2 when known)."""
    table = monthly_climatology(series)
    annual = sum(r["rain_mm"] or 0 for r in table)
    wet = [r["month"] for r in table if (r["rain_mm"] or 0) >= 60]
    return {
        "years": f'{series["dates"][0][:4]}-{series["dates"][-1][:4]}',
        "source": series.get("source"),
        "annual_rain_mm": round(annual),
        "mean_temp_c": _r(np.mean([r["tmean_c"] for r in table if r["tmean_c"] is not None])),
        "wet_months": wet,
        "monthly": table,
        "resolution_note": ("Gridded reanalysis (~9-25 km): describes the area, not your field. "
                            "Local rainfall can differ substantially."),
    }


def _r(v, nd=1):
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else round(float(v), nd)


# ---------------------------------------------------------------------------
# Forecast + advisories
# ---------------------------------------------------------------------------

def get_forecast(lat: float, lon: float, days: int = 7) -> dict:
    days = max(1, min(int(days), 16))
    key = f"fc:{round(lat, 2)}:{round(lon, 2)}:{days}"
    hit = _cache_get(key, FORECAST_TTL_S)
    if hit:
        fc = {**hit["payload"], "cached": True}
    else:
        try:
            fc = _fetch_open_meteo_forecast(lat, lon, days)
            _cache_put(key, fc)
            fc["cached"] = False
        except (WeatherUnavailable, requests.RequestException, ValueError) as e:
            stale = _cache_get(key, FORECAST_TTL_S, allow_stale=True)
            if not stale:
                raise WeatherUnavailable(f"Forecast unavailable: {e}")
            fc = {**stale["payload"], "cached": True, "stale": True}
    return {**fc, **forecast_advisories(fc)}


def forecast_advisories(fc: dict) -> dict:
    """Plain-language, rule-based flags. Prompts to think, not agronomic orders."""
    rain = [0.0 if v is None else v for v in fc.get("rain", [])]
    tmax = [v for v in fc.get("tmax", []) if v is not None]
    total = round(sum(rain), 1)
    advisories = []
    if any(r >= HEAVY_RAIN_MM for r in rain):
        advisories.append("Heavy rain day(s) forecast (>= 50 mm): check drainage and erosion risk; "
                          "avoid fertilizer/spray right before.")
    if tmax and max(tmax) >= HEAT_TMAX_C:
        advisories.append("Hot days forecast (>= 35 C): heat/water stress risk, especially at flowering.")
    if len(rain) >= 5 and total < 5:
        advisories.append("Little to no rain in the forecast: plan irrigation or hold off planting/top-dressing "
                          "that needs moisture.")
    elif len(rain) >= 3 and rain[0] + rain[1] + rain[2] >= 20:
        advisories.append("Good rain expected in the next 3 days: a decent window for planting or "
                          "top-dressing if soils are workable.")
    return {"total_rain_mm": total, "rainy_days": int(sum(r >= RAINY_DAY_MM for r in rain)),
            "max_temp_c": max(tmax) if tmax else None, "advisories": advisories}
