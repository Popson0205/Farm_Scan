import numpy as np
import pytest
from rasterio.transform import from_origin

import monitoring as m
import zones as zones_mod

# 10 m pixels in a metric CRS (UTM 32N, Abuja)
TRANSFORM = from_origin(300000, 1000000, 10, 10)
CRS = "EPSG:32632"


def make_field(seed=1, size=200, base=0.72, noise=0.015):
    rng = np.random.default_rng(seed)
    return base + rng.normal(0, noise, (size, size)).astype(np.float32)


# ------------------------------ problem areas ------------------------------

def test_problem_area_found_and_ranked_by_impact():
    arr = make_field()
    arr[40:70, 40:70] = 0.40      # big stressed patch  (30x30 px = 9 ha)
    arr[150:156, 150:156] = 0.45  # small stressed patch (6x6 px  = 0.36 ha)
    res = m.find_problem_areas(arr, TRANSFORM, CRS, "ndvi")
    areas = res["areas"]
    assert len(areas) == 2
    assert areas[0]["rank"] == 1 and areas[0]["area_ha"] == pytest.approx(9.0, rel=0.05)
    assert areas[0]["priority"] == "high"
    assert areas[0]["area_ha"] > areas[1]["area_ha"]
    # centroid of the big patch must fall inside it: UTM easting 300000+55*10, northing 1000000-55*10
    # -> check it is in lon/lat near Abuja-ish zone 32 (6E-12E), not left in metres
    c = areas[0]["centroid"]
    assert 6.0 < c["lng"] < 12.0 and 0.0 < c["lat"] < 15.0
    assert res["summary"]["check_first"]
    assert res["summary"]["flagged_pct_of_field"] == pytest.approx(
        100 * (9.0 + 0.36) / 400.0, rel=0.1)


def test_uniform_field_flags_nothing():
    res = m.find_problem_areas(make_field(seed=7), TRANSFORM, CRS, "ndvi")
    assert res["areas"] == []


def test_tiny_patch_below_min_area_ignored():
    arr = make_field()
    arr[10:13, 10:13] = 0.3       # 3x3 px = 0.09 ha = 900 m2, kept at default 400
    kept = m.find_problem_areas(arr, TRANSFORM, CRS, "ndvi")["areas"]
    dropped = m.find_problem_areas(arr, TRANSFORM, CRS, "ndvi", min_area_m2=2000)["areas"]
    assert len(kept) == 1 and dropped == []


def test_clouded_pixels_are_not_treated_as_stress():
    arr = make_field()
    arr[40:70, 40:70] = np.nan    # masked cloud
    assert m.find_problem_areas(arr, TRANSFORM, CRS, "ndvi")["areas"] == []


def test_rejects_water_index():
    with pytest.raises(ValueError):
        m.find_problem_areas(make_field(), TRANSFORM, CRS, "ndwi")


def test_geographic_crs_pixel_area():
    t = from_origin(7.40, 9.08, 0.0001, 0.0001)   # ~11 m pixels
    a = m.pixel_area_m2(t, "EPSG:4326", (100, 100))
    assert 100 < a < 140


# ------------------------------ trend / alerts ------------------------------

def obs(date, mean, valid=1.0):
    return {"obs_date": date, "source": "live", "mean": mean, "p10": mean - .1, "p90": mean + .1,
            "valid_fraction": valid, "cloud_cover": 5}


def test_decline_raises_alert():
    t = m.analyze_trend([obs("2026-07-01", .60), obs("2026-07-16", .66), obs("2026-08-01", .67),
                         obs("2026-08-16", .58)])
    assert t["status"] == "declining"
    assert [a["obs_date"] for a in t["alerts"]] == ["2026-08-16"]
    assert t["series"][-1]["change_vs_prev"] == pytest.approx(-0.09)


def test_stable_and_improving():
    assert m.analyze_trend([obs("2026-07-01", .60), obs("2026-07-16", .61)])["status"] == "stable"
    assert m.analyze_trend([obs("2026-07-01", .50), obs("2026-07-16", .60)])["status"] == "improving"


def test_cloudy_observation_excluded():
    t = m.analyze_trend([obs("2026-07-01", .70), obs("2026-07-16", .20, valid=0.3),
                         obs("2026-08-01", .69)])
    assert t["excluded_count"] == 1 and t["usable_count"] == 2
    assert t["alerts"] == []              # the bogus cloud-driven 0.20 must not raise a drop alert
    assert t["status"] == "stable"


def test_single_observation_is_insufficient():
    assert m.analyze_trend([obs("2026-07-01", .6)])["status"] == "insufficient_data"


# ------------------------------ plot ranking ------------------------------

def test_plot_ranking_flags_weak_plot():
    plots = [{"plot_id": i, "mean": v} for i, v in
             enumerate([.70, .71, .69, .70, .72, .70, .71, .45])]
    plots.append({"plot_id": 99, "mean": None})
    r = m.rank_plots(plots)
    weakest = r["plots"][0]
    assert weakest["plot_id"] == 7 and weakest["flag"] == "underperforming"
    assert r["summary"]["unranked_no_data"] == 1
    assert r["summary"]["headroom_median_to_top_quartile"] >= 0


def test_plot_ranking_needs_three_plots():
    with pytest.raises(ValueError):
        m.rank_plots([{"plot_id": 1, "mean": .5}, {"plot_id": 2, "mean": .6}])


# ------------------------------ rate plan ------------------------------

def _zones(counts):
    return [{"zone": i, "pixel_count": c} for i, c in enumerate(counts)]


def test_compensate_gives_weak_zone_more():
    z = _zones([100, 100, 100])
    plan = zones_mod.build_rate_plan(z, 100.0, base_rate=90, strategy="compensate", spread=0.2)
    assert [x["rate"] for x in z] == [108.0, 90.0, 72.0]
    assert plan["difference"] == pytest.approx(0.0, abs=1e-6)   # equal areas -> redistribution only


def test_capitalize_reverses():
    z = _zones([100, 100, 100])
    zones_mod.build_rate_plan(z, 100.0, base_rate=90, strategy="capitalize", spread=0.2)
    assert [x["rate"] for x in z] == [72.0, 90.0, 108.0]


def test_unequal_areas_change_total_honestly():
    z = _zones([300, 100, 100])   # most of the field is the weak zone -> more total product
    plan = zones_mod.build_rate_plan(z, 100.0, base_rate=90, strategy="compensate", spread=0.2)
    assert plan["difference"] > 0 and plan["variable_rate_total"] > plan["flat_rate_total"]
    assert plan["total_area_ha"] == pytest.approx(5.0)


def test_rate_plan_validation():
    with pytest.raises(ValueError):
        zones_mod.build_rate_plan(_zones([1, 1, 1]), 100.0, base_rate=0)
    with pytest.raises(ValueError):
        zones_mod.build_rate_plan(_zones([1, 1, 1]), 100.0, base_rate=10, strategy="nope")
    with pytest.raises(ValueError):
        zones_mod.build_rate_plan(_zones([1, 1, 1]), 100.0, base_rate=10, spread=0.9)


# ------------------------------ zones regression ------------------------------

def test_zone_pixel_counts_are_real_pixels_not_polygon_fragments():
    rng = np.random.default_rng(3)
    arr = rng.uniform(0.2, 0.8, (60, 60)).astype(np.float32)     # speckled -> many fragments per zone
    arr[:5, :5] = np.nan
    za = zones_mod.classify_zones(arr, n_zones=3)
    polys = zones_mod.zones_to_polygons(za, TRANSFORM, CRS, 3)
    assert sum(z["pixel_count"] for z in polys) == int((~np.isnan(arr)).sum())
