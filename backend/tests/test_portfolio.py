"""Many farms: the portfolio, farms brought in from Geohub, scouting tasks to Geohub and the answers back."""
import base64
import json
import uuid
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient

import identity
import main
from conftest import CURRENT

client = TestClient(main.app)
client.cookies.set("farm_at", "token")
SQ = lambda x, y, d=0.004: {"type": "Polygon", "coordinates": [[[x, y], [x + d, y], [x + d, y + d], [x, y + d], [x, y]]]}  # noqa: E731
ENUM_FORM, SCOUT_FORM, PROJECT = str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4())


class FakeGeohub:
    def __init__(self):
        self.enum, self.scout = [], []

    def data(self, token, path, params=None):
        assert token == "token"
        if path == "projects":
            return [{"id": PROJECT, "name": "Kuje cooperative", "forms": [
                {"id": ENUM_FORM, "name": "Farms", "template": "farm_enumeration", "features": len(self.enum)},
                {"id": SCOUT_FORM, "name": "Scouting", "template": "farm_scouting", "features": len(self.scout)},
                {"id": str(uuid.uuid4()), "name": "Plots", "template": "land_enumeration", "features": 0}]}]
        if path == f"forms/{ENUM_FORM}/features":
            return {"features": self.enum, "next": None}
        if path == f"forms/{SCOUT_FORM}/features":
            return {"features": self.scout, "next": None}
        raise identity.GeoEstateIDError("Form not found.", 404, "not_found")


@pytest.fixture
def geohub(monkeypatch, as_user, other_org):
    g = FakeGeohub()
    monkeypatch.setattr(identity, "gid", g)
    as_user(org_id=other_org, limits={"hectares": 100})
    return g


def rec(seq, geom, status="submitted", **props):
    return {"id": str(uuid.uuid4()), "seq": seq, "status": status, "geometry": geom, "properties": props, "created_by_name": "Musa"}


def test_farms_come_in_from_geohub(geohub):
    geohub.enum = [
        rec(1, SQ(7.40, 9.07), farmer_name="Hauwa Musa", community="Kuje", crops=["Maize", "Soybean"], farmer_phone="0803"),
        rec(2, SQ(7.41, 9.07), status="draft", farmer_name="Draft"),
        rec(3, SQ(7.42, 9.07), status="rejected", farmer_name="Sent back"),
        rec(4, {"type": "Point", "coordinates": [7.4, 9.0]}, farmer_name="No area"),
        rec(5, {"type": "Polygon", "coordinates": [[[7.43, 9.07], [7.44, 9.08], [7.44, 9.07], [7.43, 9.08], [7.43, 9.07]]]}, farmer_name="Bow tie"),
        rec(6, SQ(7.50, 9.07, 0.05), farmer_name="Huge"),            # ~3,000 ha: over the 100 ha plan
    ]
    forms = client.get("/api/geohub/forms", params={"template": "farm_enumeration"}).json()
    assert [f["form_id"] for f in forms] == [ENUM_FORM]
    r = client.post("/api/enumeration/import", data={"form_id": ENUM_FORM}).json()
    assert [a["name"] for a in r["added"]] == ["Hauwa Musa's farm, Kuje"]
    probs = {p["farm"]: p["problem"] for p in r["problems"]}
    assert "area" in probs["No area's farm"] and "crosses" in probs["Bow tie's farm"] and "100" in probs["Huge's farm"]
    b = client.get(f"/api/boundary/{r['added'][0]['boundary_id']}").json()
    assert b["source"] == "geohub" and b["details"]["crops"] == ["Maize", "Soybean"] and b["details"]["collected_by"] == "Musa"
    # run again: nothing doubles
    r2 = client.post("/api/enumeration/import", data={"form_id": ENUM_FORM}).json()
    assert r2["added"] == [] and r2["already_here"] == 1
    assert client.post("/api/enumeration/import", data={"form_id": SCOUT_FORM}).status_code == 404     # not an enumeration form


def test_portfolio_ranks_the_weakest_first(geohub):
    ids = [client.post("/api/boundary", data={"name": n, "geojson": json.dumps(SQ(7.40 + i * 0.01, 9.07))}).json()["boundary_id"]
           for i, n in enumerate(["A", "B", "C"])]
    for d in ("2026-07-01", "2026-07-16"):                              # A and B get images; C has none
        for bid in ids[:2]:
            client.post("/api/imagery/process-demo", data={"boundary_id": bid, "index": "ndvi", "obs_date": d})
    p = client.get("/api/portfolio").json()
    assert p["farm_count"] == 3 and p["without_images"] == 1
    assert p["farms"][-1]["name"] == "C" and p["farms"][-1]["latest"] is None
    first, second = p["farms"][0], p["farms"][1]
    assert (first["alert"], -first["latest"]["mean"]) >= (second["alert"], -second["latest"]["mean"])
    assert first["images"] == 2 and first["geometry"]["type"] == "Polygon"
    # sample data is left out where it is switched off
    main.settings.demo_tools = False
    try:
        assert client.get("/api/portfolio").json()["without_images"] == 3
    finally:
        main.settings.demo_tools = True


def test_scouting_tasks_go_to_geohub_and_answers_come_back(geohub, as_user):
    bid = client.post("/api/boundary", data={"name": "North field", "geojson": json.dumps(SQ(7.40, 9.07, 0.008))}).json()["boundary_id"]
    client.post("/api/imagery/process-demo", data={"boundary_id": bid, "index": "ndvi", "obs_date": "2026-08-01"})
    r = client.post("/api/scouting/tasks", data={"boundary_id": bid, "sensitivity": 1.0})
    assert r.status_code == 409 and r.json()["code"] == "no_scouting_form"
    # only managers choose the form; it must be a scouting form
    as_user(org_id=CURRENT["user"]["org_id"], role="agronomist", limits={"hectares": 100})
    assert client.put("/api/settings/scouting", data={"form_id": SCOUT_FORM}).status_code == 403
    as_user(org_id=CURRENT["user"]["org_id"], limits={"hectares": 100})
    assert client.put("/api/settings/scouting", data={"form_id": ENUM_FORM}).status_code == 404
    s = client.put("/api/settings/scouting", data={"form_id": SCOUT_FORM}).json()
    assert s["scouting_form"] == SCOUT_FORM and "Kuje cooperative" in s["scouting_form_name"]

    r = client.post("/api/scouting/tasks", data={"boundary_id": bid, "sensitivity": 1.0}).json()
    tasks = r["tasks"]
    assert tasks and all(t["status"] == "open" for t in tasks)
    t0 = tasks[0]
    u = urlparse(t0["link"])
    q = parse_qs(u.query)
    assert u.path == f"/collect/{PROJECT}" and q["form"] == [SCOUT_FORM]
    fill = json.loads(base64.urlsafe_b64decode(q["fill"][0] + "==="))
    assert fill == {"farm": "North field", "task_ref": t0["ref"], "seen": t0["seen"]}
    lat, lng = map(float, q["at"][0].split(","))
    assert abs(lat - t0["lat"]) < 1e-5 and abs(lng - t0["lng"]) < 1e-5
    assert t0["ref"] in r["message"] and t0["link"] in r["message"]
    # sending again replaces the open tasks
    again = client.post("/api/scouting/tasks", data={"boundary_id": bid, "sensitivity": 1.0}).json()["tasks"]
    listed = client.get("/api/scouting/tasks", params={"boundary_id": bid}).json()
    assert {t["ref"] for t in listed["tasks"]} == {t["ref"] for t in again}

    a, b = again[0], again[-1]
    pt = {"type": "Point", "coordinates": [a["lng"], a["lat"]]}
    geohub.scout = [rec(1, pt, task_ref=a["ref"], visited="Yes", causes=["Weeds"], severity="High", photos=["m1", "m2"]),
                    rec(2, pt, status="draft", task_ref=b["ref"], visited="Yes", causes=["Pests"]),
                    rec(3, pt, task_ref="FS-NOTOURS", visited="Yes", causes=["Pests"])]
    if b["ref"] != a["ref"]:
        geohub.scout.append(rec(4, pt, task_ref=b["ref"], visited="Could not reach it"))
    out = client.post("/api/scouting/refresh").json()
    assert out["tasks_updated"] == (2 if b["ref"] != a["ref"] else 1)
    got = {t["ref"]: t for t in client.get("/api/scouting/tasks", params={"boundary_id": bid}).json()["tasks"]}
    assert got[a["ref"]]["status"] == "found" and got[a["ref"]]["findings"]["causes"] == ["Weeds"] and got[a["ref"]]["findings"]["photos"] == 2
    if b["ref"] != a["ref"]:
        assert got[b["ref"]]["status"] == "unreachable"
    assert client.post("/api/scouting/refresh").json()["tasks_updated"] == 0               # nothing new
    assert client.get("/api/portfolio").json()["farms"][0]["open_tasks"] == len(again) - out["tasks_updated"]
