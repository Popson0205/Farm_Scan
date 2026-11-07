"""The satellite pipeline end to end on real GeoTIFFs (tests/fake_satellites.py), through the API."""
import base64
import io
import json
import zipfile
from datetime import date, timedelta

import numpy as np
import pytest
import rasterio
from fastapi.testclient import TestClient
from PIL import Image

import main
from fake_satellites import FakeCatalog

client = TestClient(main.app)
LON, LAT = 7.4025, 9.0725
FARM = {"type": "Polygon", "coordinates": [[[7.400, 9.070], [7.405, 9.070], [7.405, 9.075], [7.400, 9.075], [7.400, 9.070]]]}


@pytest.fixture
def sat(tmp_path, monkeypatch):
    c = FakeCatalog(tmp_path, LON, LAT).install(monkeypatch)
    monkeypatch.setattr(main, "_submit", lambda fn, *a: fn(*a))
    return c


@pytest.fixture
def farm():
    return client.post("/api/boundary", data={"name": "Raster farm", "geojson": json.dumps(FARM)}).json()["boundary_id"]


def test_search_process_every_index_and_export(sat, farm):
    sat.add_s2(date(2026, 8, 20), cloud_cover=3)
    sat.add_s2(date(2026, 8, 25), cloud_cover=12, cloud_over_field=True)
    r = client.get("/api/imagery/search", params={"boundary_id": farm, "start_date": "2026-08-01", "end_date": "2026-08-31",
                                                   "max_cloud_cover": 30})
    assert r.status_code == 200, r.text
    scenes = r.json()["scenes"]
    assert [s["date"][:10] for s in scenes] == ["2026-08-20", "2026-08-25"] and scenes[0]["coverage_pct"] == 100
    clear, cloudy = scenes[0]["scene_id"], scenes[1]["scene_id"]

    expected = {"ndvi": season_ndvi_value(date(2026, 8, 20))}
    for index in ("ndvi", "ndre", "ndwi", "ndmi", "savi", "evi"):
        r = client.post("/api/imagery/process", data={"boundary_id": farm, "scene_id": clear, "index": index})
        assert r.status_code == 200, (index, r.text)
        d = r.json()
        assert d["stats"]["pixel_count"] > 2000 and d["valid_fraction"] > 0.99, (index, d["stats"], d["valid_fraction"])
        if index in expected:                                     # the +1000 DN offset is removed: NDVI comes back right
            assert abs(d["stats"]["mean"] - expected[index]) < 0.03, d["stats"]
        img = Image.open(io.BytesIO(base64.b64decode(d["overlay_png_base64"])))
        assert img.mode == "RGBA"
        (s, w), (n, e) = d["overlay_bounds"]
        a = np.array(img)[..., 3] > 0                               # where the overlay is drawn: the farm, to a pixel
        rows, cols = np.where(a)
        lat = lambda r: n - (r + 0.5) * (n - s) / a.shape[0]        # noqa: E731
        lng = lambda c: w + (c + 0.5) * (e - w) / a.shape[1]        # noqa: E731
        px = 0.0001                                                  # ~10 m
        assert abs(lat(rows.min()) - 9.075) < px and abs(lat(rows.max()) - 9.070) < px, (lat(rows.min()), lat(rows.max()))
        assert abs(lng(cols.min()) - 7.400) < px and abs(lng(cols.max()) - 7.405) < px, (lng(cols.min()), lng(cols.max()))

    # the cloudy scene: the cloud is masked, the image kept but flagged
    d = client.post("/api/imagery/process", data={"boundary_id": farm, "scene_id": cloudy, "index": "ndvi"}).json()
    assert d["valid_fraction"] < 0.9 and d["warning"] is None or "cloud-free" in (d["warning"] or "")

    # back to the clear one, then everything that reuses it
    client.post("/api/imagery/process", data={"boundary_id": farm, "scene_id": clear, "index": "ndvi"})
    z = client.post("/api/zones/classify", data={"boundary_id": farm, "index": "ndvi", "n_zones": 3, "base_rate": 90})
    assert z.status_code == 200, z.text
    assert len(z.json()["zones"]) == 3 and z.json()["rate_plan"]
    p = client.post("/api/monitoring/problem-areas", data={"boundary_id": farm, "index": "ndvi", "sensitivity": 1.5})
    assert p.status_code == 200, p.text
    areas = p.json()["areas"]
    assert areas, p.json()
    a = areas[0]                                                  # the weak patch is south-west of the centre
    assert a["centroid"]["lat"] < LAT and a["centroid"]["lng"] < LON

    g = client.get("/api/export/geotiff", params={"boundary_id": farm, "index": "ndvi"})
    assert g.status_code == 200
    with rasterio.MemoryFile(g.content) as m, m.open() as ds:
        assert ds.crs.to_epsg() == 32632 and ds.count == 1
    sh = client.get("/api/export/shp/zones", params={"boundary_id": farm, "index": "ndvi", "n_zones": 3})
    assert sh.status_code == 200 and any(n.endswith(".shp") for n in zipfile.ZipFile(io.BytesIO(sh.content)).namelist())
    rep = client.post("/api/reports/farm", data={"boundary_id": farm, "index": "ndvi"})
    assert rep.status_code == 200, rep.text
    assert not any("crop health" in w.lower() for w in rep.json()["warnings"]), rep.json()   # the map just run is in it
    pdf = client.get(rep.json()["pdf"])
    assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF" and len(pdf.content) > 8000
    h = client.get("/api/monitoring/history", params={"boundary_id": farm, "index": "ndvi"}).json()
    assert len(h["series"]) == 2


def season_ndvi_value(d):
    from fake_satellites import season_ndvi
    return season_ndvi(d)


def test_build_the_season_from_real_files(sat, farm):
    for i in range(36):
        d = date(2026, 4, 1) + timedelta(days=7 * i)
        sat.add_s2(d, cloud_cover=60 if date(2026, 5, 15) <= d <= date(2026, 8, 1) else 4,
                   cloud_over_field=date(2026, 5, 15) <= d <= date(2026, 8, 1))
    for i in range(21):
        sat.add_s1(date(2026, 4, 1) + timedelta(days=12 * i))
    job = client.post("/api/season/build", data={"boundary_id": farm, "start_date": "2026-04-01", "end_date": "2026-12-15"}).json()
    assert job["status"] == "done", job
    assert job["result"]["optical"]["failed"] == 0 and job["result"]["radar"]["failed"] == 0, job["result"]
    assert job["result"]["optical"]["images"] == 36 and job["result"]["radar"]["images"] == 21
    s = client.get("/api/season", params={"boundary_id": farm}).json()
    assert s["based_on"] in ("ndvi", "rvi") and s["greenup"], s
    from fake_satellites import season_ndvi
    days = [date(2026, 4, 1) + timedelta(days=i) for i in range(260)]
    v = np.array([season_ndvi(d) for d in days])
    true_up = days[int(np.where(v[:int(np.argmax(v))] < 0.2 + 0.2 * 0.55)[0][-1]) + 1]
    assert abs((date.fromisoformat(s["greenup"]["date"]) - true_up).days) <= 14, (s["greenup"], true_up)


def test_links_are_signed_when_read_not_when_found(monkeypatch):
    """A Planetary Computer signature lasts about an hour; a season build can take longer, so each band link is signed
    as it is opened."""
    import planetary_computer
    import stac_client
    seen = []
    monkeypatch.setattr(planetary_computer, "sign", lambda h: seen.append(h) or h + "?sig=1")
    assert stac_client.signed("https://sentinel2l2a01.blob.core.windows.net/x/B04.tif").endswith("?sig=1")
    assert stac_client.signed("/local/B04.tif") == "/local/B04.tif" and len(seen) == 1
