"""Soil from SoilGrids when HWSD is not installed (the usual deployment), with SoilGrids' own units."""
import json

import pytest
from fastapi.testclient import TestClient

import main
import soil
import soilgrids

# the shape of a real SoilGrids 2.0 answer (values for a sandy loam in central Nigeria)
VALUES = {"sand": (620, 10), "silt": (180, 10), "clay": (200, 10), "soc": (85, 10), "phh2o": (59, 10),
          "cec": (78, 10), "bdod": (142, 100), "nitrogen": (95, 100), "cfvo": (90, 10)}
ANSWER = {"type": "Feature", "properties": {"layers": [
    {"name": name, "unit_measure": {"d_factor": d, "mapped_units": "x", "target_units": "y"},
     "depths": [{"label": lab, "range": {}, "values": {"mean": v + (20 if name == "clay" and lab in ("30-60cm", "60-100cm") else 0)}}
                for lab in soilgrids.DEPTHS]} for name, (v, d) in VALUES.items()]}}


@pytest.fixture
def no_hwsd(monkeypatch, tmp_path):
    monkeypatch.setenv("HWSD2_RASTER", str(tmp_path / "none.bil"))
    asked = []

    class R:
        status_code = 200
        def raise_for_status(self): pass
        def json(self): return ANSWER

    def get(url, params, timeout):
        asked.append(params)
        return R()
    monkeypatch.setattr(soilgrids.requests, "get", get)
    return asked


def test_units_and_depths(no_hwsd):
    p = soil.soil_profile({"type": "Polygon", "coordinates": [[[7.40, 9.07], [7.41, 9.07], [7.41, 9.08], [7.40, 9.08], [7.40, 9.07]]]})
    t = p["topsoil_0_30cm"]
    assert p["source"] == "ISRIC SoilGrids 2.0"
    assert (t["sand_pct"], t["silt_pct"], t["clay_pct"]) == (62.0, 18.0, 20.0) and t["texture_class"] == "sandy clay loam"
    assert t["ph_water"] == 5.9 and t["organic_carbon_pct"] == 0.85 and t["cec_cmol_kg"] == 7.8
    assert t["bulk_density"] == 1.42 and t["total_n_g_kg"] == 0.95 and t["coarse_frag_pct"] == 9.0
    assert p["subsoil_30_100cm"]["clay_pct"] == 22.0
    q = dict((k, v) for k, v in no_hwsd[0] if k in ("lat", "lon"))
    assert abs(q["lat"] - 9.075) < 0.006 and abs(q["lon"] - 7.405) < 0.006


def test_suitability_works_without_hwsd(no_hwsd, monkeypatch):
    import weather
    from datetime import date, timedelta
    h = {"dates": [], "rain": [], "tmax": [], "tmin": [], "source": "test"}
    d = date(2016, 1, 1)
    while d.year < 2026:
        h["dates"].append(d.isoformat()); h["rain"].append(5.0 if 4 <= d.month <= 10 else 0.2)
        h["tmax"].append(32.0); h["tmin"].append(22.0)
        d += timedelta(days=1)
    monkeypatch.setattr(weather, "get_daily_history", lambda lat, lon, years=10, **k: h)
    c = TestClient(main.app)
    bid = c.post("/api/boundary", data={"name": "Soil farm", "geojson": json.dumps(
        {"type": "Polygon", "coordinates": [[[7.40, 9.07], [7.405, 9.07], [7.405, 9.075], [7.40, 9.075], [7.40, 9.07]]]})}).json()["boundary_id"]
    r = c.post("/api/suitability/analyze", data={"boundary_id": bid, "crops": "maize,cassava,rice"})
    assert r.status_code == 200, r.text
    ranked = r.json()
    assert len(ranked["ranking"]) == 3 and not ranked["not_scored"], ranked["not_scored"]
    assert all(x["class"] in ("S1", "S2", "S3", "N") for x in ranked["ranking"])
    assert any("SoilGrids" in c for c in ranked["caveats"])
    assert c.get("/api/soil/profile", params={"boundary_id": bid}).json()["source"] == "ISRIC SoilGrids 2.0"
