"""HTTP-level tests for the soil / weather / suitability endpoints.
Soil raster sampling and weather HTTP are stubbed, so no data files or network are needed."""
import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent))
from test_suitability import build_hwsd_db, make_series, unit   # noqa: E402

import main  # noqa: E402
import soil  # noqa: E402
import weather  # noqa: E402

client = TestClient(main.app)
POLY = {"type": "Polygon", "coordinates": [[[7.40, 9.07], [7.41, 9.07], [7.41, 9.08],
                                              [7.40, 9.08], [7.40, 9.07]]]}


@pytest.fixture(scope="module")
def bid():
    r = client.post("/api/boundary", data={"name": "Suit farm", "geojson": json.dumps(POLY)})
    return r.json()["boundary_id"]


@pytest.fixture()
def stubbed(tmp_path, monkeypatch):
    db = tmp_path / "hwsd2.sqlite"
    build_hwsd_db(db, unit(101, 100, sand=52, silt=18, clay=30, ph=5.8, oc=1.1, cec=9, drainage=3))
    monkeypatch.setenv("HWSD2_DB", str(db))
    monkeypatch.setattr(soil, "smu_histogram", lambda geom: {101: 2})
    series = make_series()
    monkeypatch.setattr(weather, "get_daily_history", lambda lat, lon, years=10, end=None: series)
    monkeypatch.setattr(weather, "get_forecast", lambda lat, lon, days=7: {
        "dates": ["2026-10-22"], "rain": [60.0], "tmax": [36.0], "source": "stub",
        **weather.forecast_advisories({"rain": [60.0], "tmax": [36.0]})})


def test_soil_profile_endpoint(bid, stubbed):
    r = client.get("/api/soil/profile", params={"boundary_id": bid})
    assert r.status_code == 200
    body = r.json()
    assert body["topsoil_0_30cm"]["texture_class"] == "sandy clay loam"
    assert body["topsoil_0_30cm"]["ph_water"] == pytest.approx(5.8)
    assert body["soil_test_needed"] is True


def test_soil_endpoint_explains_missing_dataset(bid, tmp_path, monkeypatch):
    monkeypatch.setenv("HWSD2_RASTER", str(tmp_path / "none.bil"))
    r = client.get("/api/soil/profile", params={"boundary_id": bid})
    assert r.status_code == 503 and "build_hwsd_sqlite" in r.json()["detail"]
    assert client.get("/api/data-status").json()["soil"]["ready"] is False


def test_unknown_boundary_404(stubbed):
    assert client.get("/api/soil/profile", params={"boundary_id": "nope"}).status_code == 404


def test_climate_and_forecast_endpoints(bid, stubbed):
    c = client.get("/api/weather/climate", params={"boundary_id": bid}).json()
    assert len(c["monthly"]) == 12 and c["annual_rain_mm"] > 800
    f = client.get("/api/weather/forecast", params={"boundary_id": bid}).json()
    assert f["advisories"]


def test_analyze_ranks_crops_and_returns_windows(bid, stubbed):
    r = client.post("/api/suitability/analyze", data={"boundary_id": bid})
    assert r.status_code == 200
    body = r.json()
    assert len(body["ranking"]) == 8
    top = body["ranking"][0]
    assert top["planting"]["alternatives"] and top["factors"]
    assert body["soil"]["interpretation"] and body["caveats"]


def test_analyze_filters_crops_and_validates_input(bid, stubbed):
    ok = client.post("/api/suitability/analyze", data={"boundary_id": bid, "crops": "maize, cassava",
                                                      "planting_date": "05-20"}).json()
    assert {c["crop"] for c in ok["ranking"]} == {"maize", "cassava"}
    assert client.post("/api/suitability/analyze", data={"boundary_id": bid, "crops": "bogus"}).status_code == 400
    assert client.post("/api/suitability/analyze", data={"boundary_id": bid, "planting_date": "soon"}).status_code == 400


def test_weather_outage_is_a_502_not_a_crash(bid, stubbed, monkeypatch):
    def down(*a, **k):
        raise weather.WeatherUnavailable("offline")
    monkeypatch.setattr(weather, "get_daily_history", down)
    assert client.post("/api/suitability/analyze", data={"boundary_id": bid}).status_code == 502
