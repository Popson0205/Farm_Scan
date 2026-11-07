"""
A farm's season from the satellites: an NDVI timeline from every usable Sentinel-2 image, a radar timeline from
Sentinel-1 that sees through cloud, and from them when the crop greened up, peaked and was harvested.

Why radar: from about May to September clouds hide most Nigerian farms from optical satellites, often for weeks at the
time a crop emerges. Sentinel-1 is a radar (C-band, 10 m, every 6–12 days over Nigeria) and is not stopped by cloud.
Its cross-polarised backscatter (VH) rises as a canopy builds; the radar vegetation index RVI = 4·VH / (VV + VH)
(dual-pol form) follows crop growth and falls at harvest. It is noisier than NDVI (speckle, soil moisture after rain
raises both bands), so FarmScan averages over the whole farm in linear power and smooths over time before reading it.

The season reading (phenology) is the amplitude-threshold method used by TIMESAT and the MODIS land-cover dynamics
product: green-up when the smoothed curve first rises through 20% of the season's amplitude, end when it falls back
through 50%. Planting is not seen directly: a crop shows from space 1–4 weeks after sowing, so the planting estimate
is a range before green-up. Every date comes with its uncertainty (the gap between the images around it).

These are estimates to check a farmer's account or a loan's timeline against, not records of what was done.
"""
from datetime import date, timedelta

import numpy as np

# thresholds of the amplitude method
GREENUP_FRACTION = 0.20
END_FRACTION = 0.50
# a crop cycle has to move the index at least this much, or it is not called one
MIN_AMPLITUDE = {"ndvi": 0.15, "rvi": 0.10}
# a crop is visible from space this many days after sowing (cereals and legumes in the tropics)
EMERGENCE_LAG_DAYS = (7, 28)
MIN_POINTS = 6
SMOOTH_DAYS = 15


# ---------------------------------------------------------------------------
# Radar (pure helpers)
# ---------------------------------------------------------------------------

def to_db(linear):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10.0 * np.log10(linear)


def rvi(vv: np.ndarray, vh: np.ndarray) -> np.ndarray:
    """Dual-pol radar vegetation index per pixel, from linear gamma0 VV and VH (NaN where either is missing)."""
    vv = np.where(vv > 0, vv, np.nan)
    vh = np.where(vh > 0, vh, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 4.0 * vh / (vv + vh)


def radar_summary(vv: np.ndarray, vh: np.ndarray) -> dict:
    """Farm-wide radar values: means in linear power (then dB), and the RVI of those means (robust to speckle)."""
    ok = (vv > 0) & (vh > 0) & np.isfinite(vv) & np.isfinite(vh)
    n = int(ok.sum())
    if n == 0:
        return {"pixels": 0, "vv_db": None, "vh_db": None, "rvi": None}
    mvv, mvh = float(vv[ok].mean()), float(vh[ok].mean())
    return {"pixels": n, "vv_db": round(float(to_db(mvv)), 2), "vh_db": round(float(to_db(mvh)), 2),
            "rvi": round(4.0 * mvh / (mvv + mvh), 4)}


# ---------------------------------------------------------------------------
# The season (pure)
# ---------------------------------------------------------------------------

def _days(d: str) -> int:
    return date.fromisoformat(d[:10]).toordinal()


def _iso(n: float) -> str:
    return date.fromordinal(int(round(n))).isoformat()


def smooth_daily(points: list[tuple[str, float]], window: int = SMOOTH_DAYS) -> tuple[np.ndarray, np.ndarray]:
    """Daily curve from irregular observations: a 3-point running median (drops single outliers such as a cloud edge or
    a rain-soaked radar image), linear interpolation to every day, then a centred moving average."""
    pts = sorted((_days(d), float(v)) for d, v in points)
    x = np.array([p[0] for p in pts], dtype=float)
    y = np.array([p[1] for p in pts], dtype=float)
    if len(y) >= 3:
        y = np.array([y[0]] + [float(np.median(y[i - 1:i + 2])) for i in range(1, len(y) - 1)] + [y[-1]])
    days = np.arange(x[0], x[-1] + 1)
    daily = np.interp(days, x, y)
    if window > 1 and len(daily) > window:
        k = np.ones(window) / window
        pad = window // 2
        daily = np.convolve(np.pad(daily, pad, mode="edge"), k, mode="valid")[:len(days)]
    return days, daily


def _gap_around(x_obs: np.ndarray, day: float) -> float:
    """Half the gap between the observations either side of a day: how far off a crossing found there can be."""
    before, after = x_obs[x_obs <= day], x_obs[x_obs >= day]
    if not len(before) or not len(after):
        return float("inf")
    return float(after.min() - before.max()) / 2.0


def phenology(points: list[tuple[str, float]], kind: str = "ndvi") -> dict:
    """The season from one index's observations [(date, value)]: green-up, peak and end, with uncertainty in days.

    Returns {"ok": False, "reason": …} when there is too little to say: fewer than MIN_POINTS images, or a curve
    that does not move enough to be a crop cycle."""
    pts = [(d, v) for d, v in points if v is not None and np.isfinite(v)]
    if len(pts) < MIN_POINTS:
        return {"ok": False, "kind": kind, "points": len(pts),
                "reason": f"Only {len(pts)} usable image(s); at least {MIN_POINTS} across the season are needed."}
    x_obs = np.array(sorted(_days(d) for d, _ in pts), dtype=float)
    days, curve = smooth_daily(pts)
    base, top = float(np.percentile(curve, 10)), float(curve.max())
    amp = top - base
    if amp < MIN_AMPLITUDE.get(kind, 0.15):
        return {"ok": False, "kind": kind, "points": len(pts), "amplitude": round(amp, 3),
                "reason": f"The {kind.upper()} hardly changes over these dates ({amp:.2f}): no clear crop cycle "
                          "(bare, fallow, tree crop, or a crop present all along)."}
    ipk = int(np.argmax(curve))
    up, down = base + GREENUP_FRACTION * amp, base + END_FRACTION * amp
    out = {"ok": True, "kind": kind, "points": len(pts), "first_image": _iso(x_obs[0]), "last_image": _iso(x_obs[-1]),
           "base": round(base, 3), "peak_value": round(top, 3), "amplitude": round(amp, 3),
           "peak": {"date": _iso(days[ipk]), "plus_minus_days": round(_gap_around(x_obs, days[ipk]))}}

    rising = np.where(curve[:ipk + 1] < up)[0]
    if len(rising):
        i = int(rising[-1]) + 1                                      # first day at or above the threshold
        g = days[min(i, ipk)]
        unc = _gap_around(x_obs, g)
        out["greenup"] = {"date": _iso(g), "plus_minus_days": None if unc == float("inf") else round(unc)}
        lo, hi = EMERGENCE_LAG_DAYS
        out["planting"] = {"from": _iso(g - hi - (0 if unc == float("inf") else unc)),
                           "to": _iso(g - lo + (0 if unc == float("inf") else unc)),
                           "note": f"A crop shows from space {lo}–{hi} days after sowing, so sowing is put in this range."}
    else:
        out["greenup"] = None
        out["greenup_note"] = "Already green at the first image: the crop was up before " + _iso(x_obs[0]) + "."

    falling = np.where(curve[ipk:] < down)[0]
    if len(falling):
        e = days[ipk + int(falling[0])]
        unc = _gap_around(x_obs, e)
        out["end"] = {"date": _iso(e), "plus_minus_days": None if unc == float("inf") else round(unc),
                      "meaning": "harvest, or the crop drying down"}
    else:
        out["end"] = None
        out["end_note"] = "Still green at the last image (" + _iso(x_obs[-1]) + "): not harvested yet, or no image since."
    return out


def season_reading(ndvi: list[tuple[str, float]], radar: list[tuple[str, float]], declared_planting: str | None = None) -> dict:
    """Both readings and the one to rely on: NDVI when it found a cycle with its green-up (optical is the sharper
    signal); otherwise radar, which has dates when clouds hid the farm. Compares the planting estimate with the
    planting date the farmer gave, if any."""
    o, r = phenology(ndvi, "ndvi"), phenology(radar, "rvi")

    def sharpness(p):            # how closely the reading pins down green-up (smaller is better); None = cannot
        if not p["ok"] or not p.get("greenup") or p["greenup"]["plus_minus_days"] is None:
            return None
        return p["greenup"]["plus_minus_days"]
    so, sr = sharpness(o), sharpness(r)
    if so is not None and (sr is None or so <= sr + 3):
        best = o                                      # optical is the cleaner signal when it sees the green-up as well
    elif sr is not None:
        best = r
    else:
        best = o if o["ok"] else r if r["ok"] else None
    out = {"optical": o, "radar": r, "based_on": best["kind"] if best else None}
    if best:
        out.update({k: best.get(k) for k in ("greenup", "planting", "peak", "end", "greenup_note", "end_note")})
        if best is r and so is not None:
            out["note"] = (f"Radar is used: clouds left a gap in the optical images around green-up (±{so} days against "
                           f"±{sr} from radar).")
        elif best is r and o["ok"]:
            out["note"] = "Radar is used: the optical images do not show when the crop greened up (clouds)."
        elif best is r:
            out["note"] = "Radar is used: too few cloud-free optical images."
    if declared_planting and best and best.get("planting"):
        p = best["planting"]
        d = date.fromisoformat(declared_planting[:10])
        lo, hi = date.fromisoformat(p["from"]), date.fromisoformat(p["to"])
        slack = timedelta(days=21)
        if lo - slack <= d <= hi + slack:
            out["declared_check"] = {"declared": declared_planting[:10], "matches": True,
                                     "words": "The planting date given fits what the satellites show."}
        else:
            off = (lo - d).days if d < lo else (d - hi).days
            out["declared_check"] = {"declared": declared_planting[:10], "matches": False,
                                     "words": f"The planting date given is {off} days {'before' if d < lo else 'after'} what the "
                                              "satellites show. Check it with the farmer, and on the ground."}
    return out


# ---------------------------------------------------------------------------
# Fetching (network: Microsoft Planetary Computer; mocked in tests)
# ---------------------------------------------------------------------------

S1_COLLECTION = "sentinel-1-rtc"


def radar_scenes(geometry: dict, start: str, end: str, limit: int = 60) -> list[dict]:
    """Sentinel-1 radiometrically terrain-corrected scenes covering the whole farm, oldest first, one per date."""
    import planetary_computer
    from pystac_client import Client
    from stac_client import STAC_API_URL, MIN_COVERAGE, coverage_fraction

    search = Client.open(STAC_API_URL).search(collections=[S1_COLLECTION], intersects=geometry, datetime=f"{start}/{end}",
                                              max_items=400)
    by_day: dict[str, object] = {}
    for item in search.items():
        if coverage_fraction(item.geometry, geometry) >= MIN_COVERAGE:
            by_day.setdefault((item.properties.get("datetime") or "")[:10], item)
    out = []
    for day in sorted(by_day)[-max(1, int(limit)):]:
        s = planetary_computer.sign(by_day[day])
        if "vv" in s.assets and "vh" in s.assets:
            out.append({"scene_id": s.id, "date": day, "vv": s.assets["vv"].href, "vh": s.assets["vh"].href,
                        "orbit": s.properties.get("sat:orbit_state"), "platform": s.properties.get("platform")})
    return out


def radar_arrays(scene: dict, geometry: dict) -> tuple[np.ndarray, np.ndarray]:
    """VV and VH (linear gamma0) inside the farm; NaN outside it and where there is no data."""
    import rasterio
    from rasterio.mask import mask as rio_mask
    from rasterio.warp import transform_geom

    out = []
    for key in ("vv", "vh"):
        with rasterio.open(scene[key]) as src:
            g = transform_geom("EPSG:4326", src.crs, geometry)
            arr, _ = rio_mask(src, [g], crop=True, filled=True, nodata=np.nan)
            a = arr[0].astype("float32")
            a[~np.isfinite(a) | (a <= 0)] = np.nan
            out.append(a)
    vv, vh = out
    if vv.shape != vh.shape:                      # both bands share one grid in RTC; guard anyway
        h, w = min(vv.shape[0], vh.shape[0]), min(vv.shape[1], vh.shape[1])
        vv, vh = vv[:h, :w], vh[:h, :w]
    return vv, vh
