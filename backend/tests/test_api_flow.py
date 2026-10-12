"""End-to-end through the HTTP API in demo mode (no network needed)."""
import io
import json
import zipfile

import pytest
from fastapi.testclient import TestClient

import main

client = TestClient(main.app)
POLY = {"type": "Polygon", "coordinates": [[[7.40, 9.07], [7.41, 9.07], [7.41, 9.08],
                                              [7.40, 9.08], [7.40, 9.07]]]}


@pytest.fixture(scope="module")
def bid():
    r = client.post("/api/boundary", data={"name": "Test farm", "geojson": json.dumps(POLY)})
    assert r.status_code == 200
    return r.json()["boundary_id"]


def test_auto_extraction_endpoints_are_gone(bid):
    assert client.post("/api/plots/extract-from-drone", data={"drone_file_id": "x"}).status_code in (404, 405)
    assert client.post("/api/plots/detect-boundary", data={"boundary_id": bid}).status_code in (404, 405)


def test_timeline_and_alerts(bid):
    # a season of demo images; each date seeds different synthetic data
    for d in ["2026-07-01", "2026-07-16", "2026-08-01"]:
        r = client.post("/api/imagery/process-demo", data={"boundary_id": bid, "index": "ndvi", "obs_date": d})
        assert r.status_code == 200
        body = r.json()
        assert body["observation"]["obs_date"] == d and body["valid_fraction"] == 1.0
    h = client.get("/api/monitoring/history", params={"boundary_id": bid, "index": "ndvi"}).json()
    assert [s["obs_date"] for s in h["series"]] == ["2026-07-01", "2026-07-16", "2026-08-01"]
    assert h["series"][1]["change_vs_prev"] is not None
    # re-running a date replaces it instead of duplicating
    client.post("/api/imagery/process-demo", data={"boundary_id": bid, "index": "ndvi", "obs_date": "2026-08-01"})
    h2 = client.get("/api/monitoring/history", params={"boundary_id": bid, "index": "ndvi"}).json()
    assert len(h2["series"]) == 3


def test_bad_obs_date_rejected(bid):
    r = client.post("/api/imagery/process-demo", data={"boundary_id": bid, "index": "ndvi", "obs_date": "nope"})
    assert r.status_code == 400


def test_problem_areas_and_scouting_notes(bid):
    client.post("/api/imagery/process-demo", data={"boundary_id": bid, "index": "ndvi", "obs_date": "2026-08-01"})
    r = client.post("/api/monitoring/problem-areas",
                    data={"boundary_id": bid, "index": "ndvi", "sensitivity": 1.0,
                          "create_scouting_notes": "true"})
    assert r.status_code == 200
    body = r.json()
    assert body["summary"]["check_first"]
    n = body["scouting_notes_created"]
    assert n == len(body["areas"])
    for a in body["areas"]:
        assert 9.0 < a["centroid"]["lat"] < 9.1 and 7.39 < a["centroid"]["lng"] < 7.42
    notes = client.get(f"/api/annotations/{bid}").json()["annotations"]
    assert sum(1 for x in notes if x["author"] == "FarmScan scout") == n
    # human note survives a re-run; machine notes are replaced, not duplicated
    client.post("/api/annotations", data={"boundary_id": bid, "lat": 9.075, "lng": 7.405,
                                          "comment": "my own note", "author": "Ada"})
    client.post("/api/monitoring/problem-areas",
                data={"boundary_id": bid, "index": "ndvi", "sensitivity": 1.0, "create_scouting_notes": "true"})
    notes = client.get(f"/api/annotations/{bid}").json()["annotations"]
    assert sum(1 for x in notes if x["author"] == "FarmScan scout") == n
    assert any(x["comment"] == "my own note" for x in notes)


def test_problem_areas_requires_processed_index_and_valid_index(bid):
    assert client.post("/api/monitoring/problem-areas", data={"boundary_id": bid, "index": "evi"}).status_code == 400
    client.post("/api/imagery/process-demo", data={"boundary_id": bid, "index": "ndwi"})
    assert client.post("/api/monitoring/problem-areas", data={"boundary_id": bid, "index": "ndwi"}).status_code == 400


def test_plot_performance_and_excel_still_work(bid):
    client.post("/api/imagery/process-demo", data={"boundary_id": bid, "index": "ndvi", "obs_date": "2026-08-01"})
    assert client.post("/api/plots/generate-grid", data={"boundary_id": bid, "rows": 3, "cols": 4}).status_code == 200
    z = client.post("/api/plots/zonal-stats", data={"boundary_id": bid, "index": "ndvi"})
    assert z.status_code == 200 and z.json()["plot_count"] == 12
    p = client.post("/api/plots/performance", data={"boundary_id": bid, "index": "ndvi"})
    assert p.status_code == 200
    body = p.json()
    assert body["summary"]["plots_ranked"] == 12
    assert all("z_score" in x and x["flag"] in ("underperforming", "watch", "ok", "top") for x in body["plots"])
    x = client.get("/api/export/excel/plots", params={"boundary_id": bid, "index": "ndvi"})
    assert x.status_code == 200 and x.content[:2] == b"PK"


def test_rate_plan_and_export(bid):
    client.post("/api/imagery/process-demo", data={"boundary_id": bid, "index": "ndre", "obs_date": "2026-08-01"})
    r = client.post("/api/zones/classify", data={"boundary_id": bid, "index": "ndre", "n_zones": 3,
                                                  "base_rate": 90, "unit": "kg N/ha",
                                                  "strategy": "compensate", "spread": 0.2})
    assert r.status_code == 200
    body = r.json()
    plan = body["rate_plan"]
    assert plan["unit"] == "kg N/ha" and plan["total_area_ha"] > 0
    rates = [z["rate"] for z in sorted(body["zones"], key=lambda z: z["zone"])]
    assert rates[0] > rates[1] > rates[2]                       # weakest zone gets the most
    assert sum(z["area_ha"] for z in body["zones"]) == pytest.approx(plan["total_area_ha"], rel=1e-3)
    # total field area is consistent with the boundary (about 1.1 km x 1.1 km = ~120 ha)
    assert 100 < plan["total_area_ha"] < 140
    shp = client.get("/api/export/shp/zones", params={"boundary_id": bid, "index": "ndre", "n_zones": 3})
    assert shp.status_code == 200
    names = zipfile.ZipFile(io.BytesIO(shp.content)).namelist()
    assert any(n.endswith(".dbf") for n in names)
    # without base_rate the old behaviour is unchanged
    r2 = client.post("/api/zones/classify", data={"boundary_id": bid, "index": "ndre", "n_zones": 5})
    assert r2.status_code == 200 and r2.json()["rate_plan"] is None
    # invalid strategy is a clean 400
    bad = client.post("/api/zones/classify", data={"boundary_id": bid, "index": "ndre", "n_zones": 3,
                                                    "base_rate": 90, "strategy": "nope"})
    assert bad.status_code == 400
