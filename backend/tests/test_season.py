"""The season from the satellites: phenology on known curves, radar helpers, and building a farm's season (mocked)."""
import json
from datetime import date, timedelta

import numpy as np
import pytest
from fastapi.testclient import TestClient

import main
import season
import stac_client

T0 = date(2026, 4, 1)


def crop_curve(day: date, base=0.2, top=0.75, sow=date(2026, 6, 1), peak=date(2026, 8, 10), harvest=date(2026, 10, 15)) -> float:
    """A cereal season: bare until ~3 weeks after sowing, logistic green-up to a peak, drop at harvest."""
    t = (day - sow).days
    up = 1 / (1 + np.exp(-(t - 35) / 7))                       # half-way 35 days after sowing
    down = 1 / (1 + np.exp(((day - harvest).days) / 5))
    return base + (top - base) * up * down


def obs(every=10, start=T0, end=date(2026, 12, 15), noise=0.02, seed=1, **kw):
    rng = np.random.default_rng(seed)
    out, d = [], start
    while d <= end:
        out.append((d.isoformat(), crop_curve(d, **kw) + rng.normal(0, noise)))
        d += timedelta(days=every)
    return out


def true_crossing(frac, rising, **kw):
    days = [T0 + timedelta(days=i) for i in range(260)]
    v = np.array([crop_curve(d, **kw) for d in days])
    base, top = 0.2, 0.75
    th = base + frac * (top - base)
    ipk = int(np.argmax(v))
    if rising:
        return days[int(np.where(v[:ipk] < th)[0][-1]) + 1]
    return days[ipk + int(np.where(v[ipk:] < th)[0][0])]


def test_reads_green_up_peak_and_harvest():
    p = season.phenology(obs(), "ndvi")
    assert p["ok"] and p["amplitude"] > 0.45
    g = date.fromisoformat(p["greenup"]["date"])
    assert abs((g - true_crossing(0.2, True)).days) <= 10 and p["greenup"]["plus_minus_days"] <= 5
    e = date.fromisoformat(p["end"]["date"])
    assert abs((e - true_crossing(0.5, False)).days) <= 10
    pk = date.fromisoformat(p["peak"]["date"])
    assert abs((pk - date(2026, 8, 25)).days) <= 25
    lo, hi = date.fromisoformat(p["planting"]["from"]), date.fromisoformat(p["planting"]["to"])
    assert lo <= date(2026, 6, 1) <= hi and (hi - lo).days <= 40


def test_says_so_when_it_cannot_tell():
    flat = [((T0 + timedelta(days=10 * i)).isoformat(), 0.62 + 0.01 * (i % 2)) for i in range(15)]
    p = season.phenology(flat, "ndvi")
    assert not p["ok"] and "no clear crop cycle" in p["reason"]
    few = obs()[:5]
    p = season.phenology(few, "ndvi")
    assert not p["ok"] and "at least 6" in p["reason"]
    late = season.phenology(obs(start=date(2026, 7, 20)), "ndvi")        # first image after green-up
    assert late["ok"] and late["greenup"] is None and "Already green" in late["greenup_note"]
    early = season.phenology(obs(end=date(2026, 9, 20)), "ndvi")        # last image before harvest
    assert early["ok"] and early["end"] is None and "not harvested yet" in early["end_note"]


def test_radar_helpers():
    vv = np.array([[0.1, 0.2], [np.nan, 0.0]], dtype=float)
    vh = np.array([[0.02, 0.05], [0.03, 0.01]], dtype=float)
    r = season.rvi(vv, vh)
    assert abs(r[0, 0] - 4 * 0.02 / 0.12) < 1e-9 and np.isnan(r[1, 0]) and np.isnan(r[1, 1])
    s = season.radar_summary(vv, vh)
    assert s["pixels"] == 2 and s["vv_db"] == round(10 * np.log10(0.15), 2) and abs(s["rvi"] - 4 * 0.035 / 0.185) < 1e-3


def test_radar_steps_in_when_clouds_hide_the_green_up():
    # optical: clear only in the dry season (before May, after mid-October): no green-up visible
    optical = [(d, v) for d, v in obs(every=5) if d < "2026-05-01" or d > "2026-10-20"]
    radar = [(d, 0.25 + 0.5 * (v - 0.2) / 0.55) for d, v in obs(every=12, noise=0.03, seed=3)]
    r = season.season_reading(optical, radar, declared_planting="2026-03-01")
    assert r["based_on"] == "rvi" and "Radar is used" in r["note"]
    g = date.fromisoformat(r["greenup"]["date"])
    assert abs((g - true_crossing(0.2, True)).days) <= 14
    assert r["declared_check"]["matches"] is False and "before" in r["declared_check"]["words"]
    ok = season.season_reading(obs(), radar, declared_planting="2026-06-05")
    assert ok["based_on"] == "ndvi" and ok["declared_check"]["matches"] is True


# ---------------------------------------------------------------------------
# Building a farm's season through the API (satellite calls mocked)
# ---------------------------------------------------------------------------

client = TestClient(main.app)
POLY = {"type": "Polygon", "coordinates": [[[7.40, 9.07], [7.405, 9.07], [7.405, 9.075], [7.40, 9.075], [7.40, 9.07]]]}


@pytest.fixture
def satellites(monkeypatch):
    days = [T0 + timedelta(days=5 * i) for i in range(52)]
    cloudy = lambda d: date(2026, 5, 1) <= d <= date(2026, 10, 20)  # noqa: E731
    monkeypatch.setattr(stac_client, "covering_scenes", lambda g, s, e, **k: [
        {"scene_id": f"S2-{d}", "date": d.isoformat(), "cloud_cover": 80 if cloudy(d) else 5, "hrefs": {}, "offset": 0.0} for d in days])

    def bands(hrefs, geom, needed, reflectance_offset=0.0):
        d = date.fromisoformat(bands.calls.pop(0))
        ndvi = crop_curve(d)
        red = np.full((20, 20), 0.1, dtype=np.float32)
        nir = (red * (1 + ndvi) / (1 - ndvi)).astype(np.float32)
        vf = 0.1 if cloudy(d) else 0.95
        if vf < 0.5:
            red[:18], nir[:18] = np.nan, np.nan
        return {"red": red, "nir": nir}, {"valid_fraction": vf}
    bands.calls = [d.isoformat() for d in days]
    monkeypatch.setattr(stac_client, "fetch_and_clip_bands", bands)
    rdays = [T0 + timedelta(days=12 * i) for i in range(22)]
    monkeypatch.setattr(season, "radar_scenes", lambda g, s, e, **k: [{"scene_id": f"S1-{d}", "date": d.isoformat()} for d in rdays])

    def radar(scene, geom):
        v = crop_curve(date.fromisoformat(scene["date"]))
        vv = np.full((20, 20), 0.12)
        vh = vv * (0.06 + 0.25 * (v - 0.2) / 0.55)
        return vv, vh
    monkeypatch.setattr(season, "radar_arrays", radar)
    monkeypatch.setattr(main, "_submit", lambda fn, *a: fn(*a))


def test_build_the_season_and_read_it(satellites, as_user, other_org):
    bid = client.post("/api/boundary", data={"name": "Season farm", "geojson": json.dumps(POLY)}).json()["boundary_id"]
    r = client.post("/api/season/build", data={"boundary_id": bid, "start_date": "2026-04-01", "end_date": "2026-12-15"})
    assert r.status_code == 200, r.text
    job = r.json()
    assert job["status"] == "done" and job["progress"] == 100, job
    assert job["result"]["optical"]["images"] == 52 and job["result"]["radar"]["images"] == 22
    s = client.get("/api/season", params={"boundary_id": bid}).json()
    assert s["cloudy_images"] > 20 and s["based_on"] == "rvi"                      # clouds hid the green-up from optical
    g = date.fromisoformat(s["greenup"]["date"])
    assert abs((g - true_crossing(0.2, True)).days) <= 14
    assert s["end"] and abs((date.fromisoformat(s["end"]["date"]) - true_crossing(0.5, False)).days) <= 14
    assert len(s["series"]["rvi"]) == 22 and len(s["series"]["vh_db"]) == 22 and any(not p["used"] for p in s["series"]["ndvi"])
    # checks: dates, other organizations
    assert client.post("/api/season/build", data={"boundary_id": bid, "start_date": "2026-05-01", "end_date": "2026-04-01"}).status_code == 400
    assert client.post("/api/season/build", data={"boundary_id": bid, "start_date": "2024-01-01", "end_date": "2026-01-01"}).status_code == 400
    as_user(org_id=other_org)
    assert client.get(f"/api/season/jobs/{job['id']}").status_code == 404
    assert client.get("/api/season", params={"boundary_id": bid}).status_code == 404


def test_radar_wins_when_clouds_leave_a_gap_at_green_up():
    # optical sees the start and the end of the season, but nothing from May to mid-August
    optical = [(d, v) for d, v in obs(every=5) if not ("2026-05-01" <= d <= "2026-08-10")]
    radar = [(d, 0.25 + 0.5 * (v - 0.2) / 0.55) for d, v in obs(every=12, noise=0.03, seed=3)]
    r = season.season_reading(optical, radar)
    assert r["optical"]["ok"] and r["optical"]["greenup"]["plus_minus_days"] > 30
    assert r["based_on"] == "rvi" and "gap" in r["note"] and r["greenup"]["plus_minus_days"] <= 6
