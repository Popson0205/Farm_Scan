"""FarmScan on the GeoEstate platform: organizations, roles, the plan's features and hectares, files, sign-in."""
import io
import json

import numpy as np
import pytest
import rasterio
from fastapi.testclient import TestClient
from rasterio.transform import from_origin

import identity
import main
from config import settings
from conftest import CURRENT, make_user

client = TestClient(main.app)
# about 1.1 km x 1.1 km near Abuja (~121 ha)
POLY = {"type": "Polygon", "coordinates": [[[7.40, 9.07], [7.41, 9.07], [7.41, 9.08], [7.40, 9.08], [7.40, 9.07]]]}
SMALL = {"type": "Polygon", "coordinates": [[[7.400, 9.070], [7.401, 9.070], [7.401, 9.071], [7.400, 9.071], [7.400, 9.070]]]}


def add(geom=POLY, name="Farm"):
    return client.post("/api/boundary", data={"name": name, "geojson": json.dumps(geom)})


def test_needs_sign_in():
    CURRENT["user"] = None
    r = client.get("/api/boundaries")
    assert r.status_code == 401 and r.json()["code"] == "unauthenticated"
    assert client.get("/api/health").status_code == 200 and client.get("/api/indices").status_code == 200


def test_area_is_on_the_ellipsoid():
    r = add(SMALL).json()
    # 0.001° x 0.001° at 9.07°N: 110.6 m x 109.9 m = 1.2156 ha (geodesic); the old flat approximation gave 1.2234
    assert abs(r["area_ha"] - 1.22) < 0.01


def test_farms_belong_to_their_organization(as_user, other_org):
    bid = add(name="Ours").json()["boundary_id"]
    as_user(org_id=other_org, org_name="Someone Else Ltd")
    assert client.get(f"/api/boundary/{bid}").status_code == 404
    assert client.get(f"/api/plots/{bid}").status_code == 404
    assert client.post("/api/imagery/process-demo", data={"boundary_id": bid, "index": "ndvi"}).status_code == 404
    assert client.get("/api/export/geotiff", params={"boundary_id": bid, "index": "ndvi"}).status_code == 404
    assert all(b["id"] != bid for b in client.get("/api/boundaries").json())
    assert client.get("/api/boundary/not-a-uuid").status_code == 404


def test_viewers_read_only_and_expired_licences(as_user):
    bid = add().json()["boundary_id"]
    as_user(role="viewer", org_role="viewer")
    assert client.get(f"/api/boundary/{bid}").status_code == 200
    r = add()
    assert r.status_code == 403 and r.json()["code"] == "forbidden"
    as_user(licence="expired", can_write=False)
    r = client.post("/api/plots/generate-grid", data={"boundary_id": bid, "rows": 2, "cols": 2})
    assert r.status_code == 403 and r.json()["code"] == "licence_expired"
    assert client.get(f"/api/plots/{bid}").status_code == 200                    # still readable
    as_user(role="agronomist", org_role="member")
    assert client.delete(f"/api/boundary/{bid}").status_code == 403               # only managers delete farms


def test_plan_features(as_user):
    bid = add().json()["boundary_id"]
    as_user(plan="standard", features=["monitoring", "zones", "suitability"])
    client.post("/api/imagery/process-demo", data={"boundary_id": bid, "index": "ndvi"})
    r = client.post("/api/zones/classify", data={"boundary_id": bid, "index": "ndvi", "n_zones": 3})
    assert r.status_code == 200 and r.json()["rate_plan"] is None
    r = client.post("/api/zones/classify", data={"boundary_id": bid, "index": "ndvi", "n_zones": 3, "base_rate": 90})
    assert r.status_code == 403 and r.json()["code"] == "feature_not_in_plan" and "prescriptions" in r.json()["detail"]
    r = client.post("/api/drone/upload", files={"file": ("x.tif", b"nope", "image/tiff")})
    assert r.status_code == 403 and "Drone imagery" in r.json()["detail"]
    as_user(features=[])
    assert client.get("/api/monitoring/history", params={"boundary_id": bid}).status_code == 403


def test_hectares_limit(as_user, other_org):
    as_user(org_id=other_org, limits={"hectares": 200})
    assert add().status_code == 200                    # ~121 ha
    r = add()
    assert r.status_code == 403 and r.json()["code"] == "hectares_limit" and "200" in r.json()["detail"]
    me = client.get("/api/me").json()
    assert me["farms"] == 1 and 120 < me["hectares_used"] < 123 and me["hectares_limit"] == 200
    # removing a farm frees its hectares
    bid = client.get("/api/boundaries").json()[0]["id"]
    assert client.delete(f"/api/boundary/{bid}").status_code == 204
    assert add().status_code == 200


def test_sample_data_only_where_switched_on():
    bid = add().json()["boundary_id"]
    settings.demo_tools = False
    try:
        assert client.post("/api/imagery/process-demo", data={"boundary_id": bid, "index": "ndvi"}).status_code == 404
        assert client.get("/api/me").json()["demo_tools"] is False
    finally:
        settings.demo_tools = True


def _tiff() -> bytes:
    buf = io.BytesIO()
    arr = (np.random.default_rng(1).random((3, 64, 64)) * 255).astype("uint8")
    with rasterio.MemoryFile() as mem:
        with mem.open(driver="GTiff", width=64, height=64, count=3, dtype="uint8", crs="EPSG:32632",
                      transform=from_origin(330000, 1003000, 0.03, 0.03)) as ds:
            ds.write(arr)
        buf.write(mem.read())
    return buf.getvalue()


def test_drone_files_are_kept_for_the_organization(as_user, other_org):
    r = client.post("/api/drone/upload", files={"file": ("field.tif", _tiff(), "image/tiff")})
    assert r.status_code == 200, r.text
    fid = r.json()["drone_file_id"]
    assert client.post("/api/drone/plant-count", data={"drone_file_id": fid}).status_code == 200
    as_user(org_id=other_org)
    assert client.post("/api/drone/plant-count", data={"drone_file_id": fid}).status_code == 404


def test_share_links_open_without_sign_in():
    bid = add().json()["boundary_id"]
    client.post("/api/imagery/process-demo", data={"boundary_id": bid, "index": "ndvi"})
    token = client.post("/api/share", data={"boundary_id": bid, "index": "ndvi"}).json()["token"]
    CURRENT["user"] = None
    r = client.get(f"/api/share/{token}")
    assert r.status_code == 200 and r.json()["boundary"]["boundary_id"] == bid and r.json()["source"] == "demo"


def test_notes_carry_the_author():
    bid = add().json()["boundary_id"]
    n = client.post("/api/annotations", data={"boundary_id": bid, "lat": 9.075, "lng": 7.405, "comment": "Gap near the gate"}).json()
    assert n["author"] == "Ada Farmer"


def test_sign_in_goes_to_geohub_and_back(monkeypatch):
    class FakeID:
        def authorize_url(self, redirect_uri, state, verifier):
            return f"http://geohub.test/id/authorize?client_id=farm&redirect_uri={redirect_uri}&state={state}"

        def exchange(self, code, redirect_uri, verifier):
            assert code == "c0de" and verifier
            return {"access_token": "at", "refresh_token": "rt", "expires_in": 900}

    monkeypatch.setattr(identity, "gid", FakeID())
    r = client.get("/api/auth/sign-in", params={"next": "/?farm=1"}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"].startswith("http://geohub.test/id/authorize?client_id=farm")
    state = r.cookies.get("farm_state")
    r = client.get("/api/auth/callback", params={"code": "c0de", "state": state}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "http://farm.test/?farm=1"
    assert "farm_at=at" in r.headers.get("set-cookie", "")
    # a callback that did not start here is refused
    client.cookies.clear()
    r = client.get("/api/auth/callback", params={"code": "c0de", "state": "other"}, follow_redirects=False)
    assert "signin_error" in r.headers["location"]


def test_real_sign_in_checks_tokens_with_geohub(monkeypatch):
    """Without the test override: the access-token cookie is verified, roles and the plan come from it."""
    claims = {"sub": "u9", "name": "Bola", "email": "b@example.com", "org": {"id": "22222222-2222-4222-8222-222222222222",
              "name": "Coop", "role": "member"}, "product": {"plan": "standard", "status": "active",
              "features": ["monitoring"], "limits": {"hectares": 500}}}

    class FakeID:
        def verify(self, token):
            if token != "good":
                raise identity.GeoEstateIDError("bad")
            return claims

    monkeypatch.setattr(identity, "gid", FakeID())
    main.app.dependency_overrides.pop(identity.current_user)
    try:
        c = TestClient(main.app)
        assert c.get("/api/me").status_code == 401
        c.cookies.set("farm_at", "good")
        me = c.get("/api/me").json()
        assert me["role"] == "agronomist" and me["features"] == ["monitoring"] and me["hectares_limit"] == 500
        claims["product"]["status"] = "suspended"
        assert c.get("/api/me").json()["code"] == "licence_suspended"
    finally:
        from conftest import _current
        main.app.dependency_overrides[identity.current_user] = _current
