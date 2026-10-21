"""Soil aggregation, weather windows, caching and suitability scoring.
Pure logic: no network, no raster file, no FastAPI needed."""
import sqlite3
from datetime import date, timedelta

import numpy as np
import pytest

import soil
import suitability as suit
import weather


# ---------------------------------------------------------------------------
# Fixtures / synthetic data
# ---------------------------------------------------------------------------

def make_series(years=10, first_year=2015, seed=7, rain_scale=1.0):
    """Abuja-like climate: single rainy season Apr-Oct, ~1400 mm/yr, 24-30 C."""
    monthly = [0, 5, 20, 80, 150, 190, 220, 260, 280, 150, 20, 2]
    rng = np.random.default_rng(seed)
    dates, tmax, tmin, tmean, rain = [], [], [], [], []
    d = date(first_year, 1, 1)
    end = date(first_year + years - 1, 12, 31)
    while d <= end:
        dim = 30
        p = min(0.9, monthly[d.month - 1] * rain_scale / (12.0 * dim))
        r = float(rng.exponential(12.0)) if rng.random() < p else 0.0
        hot = 36 if d.month in (2, 3, 4) else 30 if d.month in (7, 8, 9) else 33
        dates.append(d.isoformat())
        tmax.append(float(hot))
        tmin.append(float(hot - 10))
        tmean.append(float(hot - 5))
        rain.append(r)
        d += timedelta(days=1)
    return {"dates": dates, "tmax": tmax, "tmin": tmin, "tmean": tmean, "rain": rain,
            "et0": [4.0] * len(dates), "source": "synthetic"}


def build_hwsd_db(path, rows):
    conn = sqlite3.connect(path)
    cols = ["HWSD2_SMU_ID", "SHARE", "LAYER", "SAND", "SILT", "CLAY", "PH_WATER", "ORG_CARBON",
            "CEC_SOIL", "ELEC_COND", "COARSE", "DRAINAGE", "WRB4", "BSAT"]
    conn.execute(f"CREATE TABLE HWSD2_LAYERS ({', '.join(c + ' ' + ('TEXT' if c in ('LAYER', 'WRB4') else 'REAL') for c in cols)})")
    for r in rows:
        conn.execute(f"INSERT INTO HWSD2_LAYERS ({','.join(r)}) VALUES ({','.join('?' * len(r))})", list(r.values()))
    conn.commit()
    conn.close()


def unit(smu, share, sand, silt, clay, ph, oc, cec, drainage=3, wrb="Lixisol", ec=0.3, coarse=5, bsat=50,
         layers=("D1", "D2", "D3", "D4", "D5", "D6", "D7")):
    return [{"HWSD2_SMU_ID": smu, "SHARE": share, "LAYER": L, "SAND": sand, "SILT": silt, "CLAY": clay,
             "PH_WATER": ph, "ORG_CARBON": oc, "CEC_SOIL": cec, "ELEC_COND": ec, "COARSE": coarse,
             "DRAINAGE": drainage, "WRB4": wrb, "BSAT": bsat} for L in layers]


def profile_for(rows, smu_pixels):
    agg = soil.aggregate_layers(rows, smu_pixels)
    return soil.profile_from_parts(agg, smu_pixels)


LOAM = dict(sand=40, silt=40, clay=20, ph=6.2, oc=1.5, cec=10)


# ---------------------------------------------------------------------------
# Scoring primitives
# ---------------------------------------------------------------------------

def test_trapezoid_shape():
    assert suit.trap(10, 5, 8, 12, 15) == 1.0
    assert suit.trap(6.5, 5, 8, 12, 15) == pytest.approx(0.5)
    assert suit.trap(13.5, 5, 8, 12, 15) == pytest.approx(0.5)
    assert suit.trap(4, 5, 8, 12, 15) == 0.0 and suit.trap(16, 5, 8, 12, 15) == 0.0
    assert suit.trap(1000, 3, 8, None, None) == 1.0          # open-ended optimum
    assert suit.trap(0, 0, 0, 15, 60) == 1.0                  # ramp-down starts at 0
    assert suit.trap(None, 0, 1, 2, 3) is None


def test_class_thresholds():
    assert suit.classify(0.85)[0] == "S1" and suit.classify(0.7)[0] == "S2"
    assert suit.classify(0.5)[0] == "S3" and suit.classify(0.1)[0] == "N"


def test_usda_texture():
    assert soil.usda_texture(40, 40, 20) == "loam"
    assert soil.usda_texture(90, 5, 5) == "sand"
    assert soil.usda_texture(20, 30, 50) == "clay"
    assert soil.usda_texture(10, 60, 30) == "silty clay loam"
    assert soil.usda_texture(65, 20, 15) == "sandy loam"


# ---------------------------------------------------------------------------
# Soil aggregation
# ---------------------------------------------------------------------------

def test_share_and_pixel_weighting():
    # SMU 101: two soil units 70/30 ; SMU 102: one unit. 3 px vs 1 px.
    rows = (unit(101, 70, 60, 20, 20, 6.0, 1.0, 8) + unit(101, 30, 30, 30, 40, 5.0, 2.0, 16)
            + unit(102, 100, 80, 10, 10, 7.0, 0.5, 4))
    agg = soil.aggregate_layers(rows, {101: 3, 102: 1})
    # weights: A=3*70=210, B=3*30=90, C=1*100=100 -> total 400
    expect_sand = (210 * 60 + 90 * 30 + 100 * 80) / 400
    assert agg["layers"]["D1"]["sand_pct"] == pytest.approx(expect_sand)
    assert agg["layers"]["D1"]["ph_water"] == pytest.approx((210 * 6.0 + 90 * 5.0 + 100 * 7.0) / 400)


def test_missing_values_renormalise_weights():
    a = unit(101, 50, 50, 30, 20, 6.0, 1.0, 8)
    b = unit(101, 50, 50, 30, 20, 6.0, -9, 8)       # negative = missing in HWSD-style data
    agg = soil.aggregate_layers(a + b, {101: 1})
    assert agg["layers"]["D1"]["organic_carbon_pct"] == pytest.approx(1.0)   # not diluted to 0.5


def test_depth_average_0_30():
    rows = unit(101, 100, 50, 30, 20, 6.0, 1.0, 8, layers=("D1",)) + \
           unit(101, 100, 50, 30, 20, 7.5, 1.0, 8, layers=("D2",))
    agg = soil.aggregate_layers(rows, {101: 1})
    top = soil.depth_average(agg["layers"], 0, 30)
    assert top["ph_water"] == pytest.approx((20 * 6.0 + 10 * 7.5) / 30)


def test_drainage_distribution_and_profile_shape():
    rows = unit(101, 60, 40, 40, 20, 6.2, 1.5, 10, drainage=3) + unit(101, 40, 40, 40, 20, 6.2, 1.5, 10, drainage=6)
    prof = profile_for(rows, {101: 2})
    assert prof["drainage_class_weights"] == {3: 0.6, 6: 0.4}
    assert prof["drainage_distribution"]["well drained"] == 0.6
    assert prof["topsoil_0_30cm"]["texture_class"] == "loam"
    assert prof["soil_test_needed"] and prof["notes"]


def test_unit_missing_from_database_is_reported(tmp_path, monkeypatch):
    db = tmp_path / "h.sqlite"
    build_hwsd_db(db, unit(101, 100, 40, 40, 20, 6.2, 1.5, 10))
    monkeypatch.setenv("HWSD2_DB", str(db))
    assert soil.load_layer_rows([999]) == []
    assert len(soil.load_layer_rows([101])) == 7


def test_status_reports_missing_files(tmp_path, monkeypatch):
    monkeypatch.setenv("HWSD2_RASTER", str(tmp_path / "nope.bil"))
    monkeypatch.setenv("HWSD2_DB", str(tmp_path / "nope.sqlite"))
    st = soil.status()
    assert not st["ready"] and not st["raster_found"]
    with pytest.raises(soil.SoilDataUnavailable):
        soil.smu_histogram({"type": "Point", "coordinates": [7, 9]})


# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------

def test_window_matches_direct_sum():
    s = make_series()
    idx = weather.WeatherIndex(s)
    w = idx.window(date(2019, 5, 1), 120)
    i0 = s["dates"].index("2019-05-01")
    assert w["rain_mm"] == pytest.approx(sum(s["rain"][i0:i0 + 120]))
    assert w["tmean_c"] == pytest.approx(np.mean(s["tmean"][i0:i0 + 120]))


def test_window_tolerates_small_gaps_but_rejects_big_ones():
    s = make_series(years=5)
    i0 = s["dates"].index("2018-05-01")
    for k in range(5):
        s["rain"][i0 + k] = None                       # 5 of 100 days missing -> 95% valid
    idx = weather.WeatherIndex(s)
    assert idx.window(date(2018, 5, 1), 100) is not None
    for k in range(40):
        s["rain"][i0 + k] = None                       # 40% missing
    assert weather.WeatherIndex(s).window(date(2018, 5, 1), 100) is None


def test_window_extras_dry_spell():
    s = make_series(years=5)
    i0 = s["dates"].index("2018-06-01")
    for k in range(12):
        s["rain"][i0 + k] = 0.0
    idx = weather.WeatherIndex(s)
    assert idx.window_extras(date(2018, 6, 1), 30)["max_dry_spell_days"] >= 12


def test_climate_summary_finds_rainy_season():
    c = weather.climate_summary(make_series())
    assert 1100 < c["annual_rain_mm"] < 1700
    assert {"Jul", "Aug", "Sep"} <= set(c["wet_months"])
    assert "Jan" not in c["wet_months"]


def test_forecast_advisories():
    fc = {"rain": [60, 0, 0, 0, 0], "tmax": [36, 33, 33, 33, 33]}
    adv = " ".join(weather.forecast_advisories(fc)["advisories"])
    assert "Heavy rain" in adv and "Hot days" in adv
    dry = weather.forecast_advisories({"rain": [0] * 7, "tmax": [30] * 7})
    assert any("Little to no rain" in a for a in dry["advisories"])


def test_history_cached_and_provider_fallback(monkeypatch):
    calls = {"om": 0, "nasa": 0}
    ser = make_series(years=4)

    def om(*a, **k):
        calls["om"] += 1
        raise weather.WeatherUnavailable("down")

    def nasa(*a, **k):
        calls["nasa"] += 1
        return {**ser, "source": "NASA POWER (daily agroclimatology)"}

    monkeypatch.setattr(weather, "_fetch_open_meteo_history", om)
    monkeypatch.setattr(weather, "_fetch_nasa_power_history", nasa)
    r1 = weather.get_daily_history(9.07, 7.40, years=4, end=date(2018, 12, 31))
    assert r1["source"].startswith("NASA") and not r1["cached"]
    r2 = weather.get_daily_history(9.07, 7.40, years=4, end=date(2018, 12, 31))
    assert r2["cached"] and calls == {"om": 1, "nasa": 1}      # second call served from cache


def test_history_raises_when_everything_is_down(monkeypatch):
    def boom(*a, **k):
        raise weather.WeatherUnavailable("down")
    monkeypatch.setattr(weather, "_fetch_open_meteo_history", boom)
    monkeypatch.setattr(weather, "_fetch_nasa_power_history", boom)
    with pytest.raises(weather.WeatherUnavailable):
        weather.get_daily_history(1.23, 4.56, years=4, end=date(2018, 12, 31))


# ---------------------------------------------------------------------------
# End-to-end suitability
# ---------------------------------------------------------------------------

def run(rows, crops=None, planting=None, series=None):
    prof = profile_for(rows, {101: 1})
    return suit.analyze(prof, series or make_series(), crops, planting)


def by(res):
    return {r["crop"]: r for r in res["ranking"]}


def test_good_loam_suits_maize_in_rainy_season():
    res = run(unit(101, 100, **LOAM, drainage=3))
    maize = by(res)["maize"]
    assert maize["class"] in ("S1", "S2")
    best = maize["planting"]["alternatives"][0]
    assert 4 <= best["start_month"] <= 7                 # starts when the rains do, not in the dry season
    assert maize["planting"]["season"]["longest_dry_spell_days_avg"] is not None


def test_acid_soil_makes_ph_the_limit_and_suggests_liming():
    res = run(unit(101, 100, **{**LOAM, "ph": 4.3}, drainage=3), crops=["maize"])
    m = by(res)["maize"]
    assert m["limiting_factor"]["key"] == "ph"
    assert any("lim" in r["advice"].lower() for r in m["recommendations"] if r["factor"] == "ph")


def test_limit_is_only_flagged_when_it_actually_limits():
    res = by(run(unit(101, 100, **{**LOAM, "oc": 2.0, "cec": 14}, drainage=3), crops=["maize"]))["maize"]
    assert res["limiting_factor"]["significant"] == (res["limiting_factor"]["score"] < 0.8)
    acid = by(run(unit(101, 100, **{**LOAM, "ph": 4.3}, drainage=3), crops=["maize"]))["maize"]
    assert acid["limiting_factor"]["significant"] is True


def test_poor_drainage_flips_rice_versus_cassava():
    rows = unit(101, 100, 20, 30, 50, 6.0, 1.5, 14, drainage=6)       # heavy, poorly drained
    res = by(run(rows, crops=["rice", "cassava"]))
    d = {r["crop"]: next(f for f in r["factors"] if f["key"] == "drainage")["score"] for r in res.values()}
    assert d["rice"] > 0.9 and d["cassava"] < 0.3
    assert res["rice"]["index"] > res["cassava"]["index"]


def test_dry_climate_penalises_rain_hungry_crops_more_than_sorghum():
    dry = make_series(rain_scale=0.35)                                # ~480 mm/yr
    res = by(run(unit(101, 100, **LOAM, drainage=3), crops=["rice", "sorghum"], series=dry))
    rain = {c: next(f for f in r["factors"] if f["key"] == "rain")["score"] for c, r in res.items()}
    assert rain["rice"] < rain["sorghum"]
    assert rain["rice"] < 0.2


def test_user_specified_planting_date_is_respected():
    res = by(run(unit(101, 100, **LOAM, drainage=3), crops=["maize"], planting="2026-06-01"))
    assert res["maize"]["planting"]["chosen_start"] == "1 Jun"
    assert suit.analyze  # smoke
    with pytest.raises(ValueError):
        run(unit(101, 100, **LOAM, drainage=3), planting="June first")


def test_planting_before_the_rains_fails_on_establishment():
    rows = unit(101, 100, **LOAM, drainage=3)
    early = by(run(rows, crops=["maize"], planting="01-15"))["maize"]      # mid dry season
    est = next(f for f in early["factors"] if f["key"] == "establish")
    assert est["score"] < 0.1
    assert early["class"] == "N"                       # several factors are 0 here, so don't assert which is 'the' limit
    assert any(r["factor"] == "establish" for r in early["recommendations"])
    during = by(run(rows, crops=["maize"], planting="05-20"))["maize"]
    assert next(f for f in during["factors"] if f["key"] == "establish")["score"] > 0.9


def test_best_windows_follow_the_rains_for_every_crop():
    res = run(unit(101, 100, **LOAM, drainage=3))
    for r in res["ranking"]:
        best = r["planting"]["alternatives"][0]
        assert 3 <= best["start_month"] <= 9, (r["crop"], best["plant_from"])     # never Nov-Feb in this climate


def test_unknown_crop_rejected():
    with pytest.raises(ValueError):
        run(unit(101, 100, **LOAM), crops=["unobtainium"])


def test_ranking_orders_by_class_then_index():
    res = run(unit(101, 100, **LOAM, drainage=3))
    order = [(suit.CLASS_ORDER[r["class"]], -r["index"]) for r in res["ranking"]]
    assert order == sorted(order)


def test_too_little_data_is_reported_not_guessed():
    rows = [{"HWSD2_SMU_ID": 101, "SHARE": 100, "LAYER": "D1", "SAND": 40, "SILT": 40, "CLAY": 20}]
    prof = profile_for(rows, {101: 1})
    short = make_series(years=2)                 # < MIN_YEARS_REQUIRED -> no climate factors
    res = suit.analyze(prof, short, ["maize"])
    assert res["ranking"] == [] and "error" in res["not_scored"][0]


def test_crop_table_is_well_formed():
    crops = suit.load_crops()
    assert len(crops) >= 8
    for cid, c in crops.items():
        assert c["cycle_days"] > 0
        for key, (a, b, cc, d) in c["factors"].items():
            assert key in suit.FACTOR_LABELS
            hi_c = float("inf") if cc is None else cc
            hi_d = float("inf") if d is None else d
            assert a <= b <= hi_c <= hi_d, (cid, key)
        assert set(c["drainage"]) == {str(i) for i in range(1, 8)}
