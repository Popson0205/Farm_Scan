"""Weekly farm alerts and farm verification reports (weather, fires and Land Registry mocked)."""
import json
import os
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pytest
from fastapi.testclient import TestClient

import alerts
import mailer
import main
import monitoring
import weather
from conftest import CURRENT

client = TestClient(main.app)
POLY = {"type": "Polygon", "coordinates": [[[7.40, 9.07], [7.404, 9.07], [7.404, 9.074], [7.40, 9.074], [7.40, 9.07]]]}
TODAY = date(2026, 7, 20)


def wx(rain_past, rain_ahead, tmax_ahead=None):
    days = len(rain_past) + len(rain_ahead)
    dates = [(TODAY - timedelta(days=len(rain_past) - 1) + timedelta(days=i)).isoformat() for i in range(days)]
    return {"dates": dates, "rain": rain_past + rain_ahead, "tmax": [32.0] * len(rain_past) + (tmax_ahead or [33.0] * len(rain_ahead)),
            "today_index": len(rain_past) - 1}


def test_weather_alerts():
    wet = [5, 6, 7, 8, 9]
    dry = alerts.weather_alerts(wx([12.0] + [0.0] * 13, [0.5] * 7), wet, TODAY)
    assert [a["kind"] for a in dry] == ["dry_spell"] and "13 days" in dry[0]["words"]
    assert alerts.weather_alerts(wx([12.0] + [0.0] * 13, [0.5] * 7), [11, 12, 1], TODAY) == []     # dry season: expected
    assert alerts.weather_alerts(wx([0.0] * 14, [3.0, 4.0, 0, 0, 0, 0, 0]), wet, TODAY) == []      # rain coming
    wet_week = alerts.weather_alerts(wx([5.0] * 14, [10, 62, 5, 0, 0, 0, 0], [33, 34, 39, 33, 33, 33, 33]), wet, TODAY)
    kinds = {a["kind"]: a["words"] for a in wet_week}
    assert "62 mm" in kinds["heavy_rain"] and "39 °C" in kinds["heat"] and "dry_spell" not in kinds


def test_fire_alerts():
    on = {"latitude": 9.072, "longitude": 7.402, "acq_date": "2026-07-18"}
    near = {"latitude": 9.072, "longitude": 7.4075, "acq_date": "2026-07-17"}          # ~390 m east of the edge
    far = {"latitude": 9.10, "longitude": 7.45, "acq_date": "2026-07-18"}
    a = alerts.fire_alerts([on, near, far], POLY)
    assert a[0]["level"] == "high" and "1 on the farm and 1 within 500 m" in a[0]["words"]
    assert alerts.fire_alerts([far], POLY) == []
    assert alerts.fire_alerts([near], POLY)[0]["level"] == "medium"


@pytest.fixture
def mocked(monkeypatch, as_user, other_org):
    os.environ["MAIL_TEST_OUTBOX"] = "1"
    mailer.OUTBOX.clear()
    monkeypatch.setattr(weather, "get_recent_and_ahead", lambda lat, lon, **k: wx([15.0] + [0.0] * 13, [0.0] * 7))
    hist = {"dates": [], "rain": [], "source": "ERA5 (test)"}
    d = date(2016, 1, 1)
    while d.year < 2026:
        hist["dates"].append(d.isoformat()); hist["rain"].append(4.0 if 4 <= d.month <= 10 else 0.2)
        d += timedelta(days=1)
    monkeypatch.setattr(weather, "get_daily_history", lambda lat, lon, years=10, **k: hist)
    monkeypatch.setattr(weather, "climate_summary", lambda h: {"wet_months": list(range(4, 11))})
    monkeypatch.setattr(alerts, "fetch_fires", lambda g, **k: [{"latitude": 9.072, "longitude": 7.402, "acq_date": "2026-07-18"}])
    as_user(org_id=other_org, org_name="Kuje Growers")
    yield
    os.environ.pop("MAIL_TEST_OUTBOX", None)


def test_alerts_page_and_weekly_email(mocked, as_user):
    bid = client.post("/api/boundary", data={"name": "North field", "geojson": json.dumps(POLY)}).json()["boundary_id"]
    for d, v in (("2026-07-01", 0.62), ("2026-07-16", 0.48)):              # NDVI fell 0.14 in the last image
        monitoring.save_observation(bid, "ndvi", d, "live", np.full((4, 4), v), valid_fraction=0.95)
    wk = main.org_alerts(CURRENT["user"]["org_id"], TODAY)                 # as on 20 July, in the rainy season
    kinds = {x["kind"]: x["words"] for f in wk["farms"] for x in f["alerts"]}
    assert set(kinds) == {"dry_spell", "fire", "crop_drop"} and "down 0.14" in kinds["crop_drop"] and wk["unavailable"] == []
    a = client.get("/api/alerts").json()
    assert a["farms_checked"] == 1 and any(x["kind"] == "fire" for f in a["farms"] for x in f["alerts"])
    assert a["subscribed"] is False and a["email_set_up"] is True
    assert client.post("/api/alerts/subscribe").json() == {"subscribed": True, "email": "ada@example.com"}
    org = CURRENT["user"]["org_id"]
    sent = main.send_due_alerts()
    assert org in sent
    mail = [m for m in mailer.OUTBOX if m["to"] == "ada@example.com"]
    assert len(mail) == 1 and "Kuje Growers" in mail[0]["subject"] and "North field" in mail[0]["text"] and "Fire:" in mail[0]["text"]
    assert org not in main.send_due_alerts()                                # not again within the week
    assert org in main.send_due_alerts(datetime.now(timezone.utc) + timedelta(days=7))
    # a licence without alerts, or expired, gets none
    as_user(org_id=org, org_name="Kuje Growers", licence="expired", can_write=False)
    client.get("/api/me")
    assert org not in main.send_due_alerts(datetime.now(timezone.utc) + timedelta(days=14))
    assert client.delete("/api/alerts/subscribe").json() == {"subscribed": False}
    as_user(org_id=org, features=["monitoring"])
    assert client.get("/api/alerts").status_code == 403


def test_verification_report(mocked, as_user):
    bid = client.post("/api/boundary", data={"name": "Loan farm", "geojson": json.dumps(POLY)}).json()["boundary_id"]
    from test_season import crop_curve
    for i in range(45):
        d = date(2026, 4, 1) + timedelta(days=6 * i)
        monitoring.save_observation(bid, "ndvi", d.isoformat(), "live", np.full((4, 4), crop_curve(d)), valid_fraction=0.95)
    as_user(org_id=CURRENT["user"]["org_id"], org_name="Kuje Growers", features=["monitoring"])
    assert client.post("/api/reports/verification", data={"boundary_id": bid}).status_code == 403       # Advanced only
    as_user(org_id=CURRENT["user"]["org_id"], org_name="Kuje Growers")
    r = client.post("/api/reports/verification", data={"boundary_id": bid})
    assert r.status_code == 200, r.text
    rep = r.json()
    assert rep["code"].startswith("FSV-") and rep["seasons"] == 1
    assert [w for w in rep["warnings"] if "crop health" not in w.lower()] == []   # no index was run
    pdf = client.get(rep["pdf"])
    assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF" and len(pdf.content) > 3000
    listed = client.get("/api/reports", params={"boundary_id": bid}).json()
    assert listed[0]["code"] == rep["code"] and listed[0]["farm"] == "Loan farm"
    # anyone can check it, without signing in; no farmer details leave
    CURRENT["user"] = None
    v = client.get(f"/api/verify/{rep['code'].lower()}").json()
    assert v["valid"] and v["issued_to"] == "Kuje Growers" and v["farm"] == "Loan farm" and v["seasons"] == [2026]
    assert client.get("/api/verify/FSV-NOTREAL1").json()["valid"] is False
    assert client.get("/verify.html").status_code == 200


def test_report_content_reads_the_season(mocked):
    import reports
    rf = reports.rainfall(weather.get_daily_history(0, 0), {"wet_months": [4, 5]})
    assert rf["normal_mm"] > 800 and len(rf["years"]) == 3 and rf["years"][-1]["year"] == 2025 and rf["years"][-1]["vs_normal_pct"] == 0


def test_brevo_is_used_like_geohub(monkeypatch):
    """With BREVO_API_KEY set (as on Geohub), the alert email goes through Brevo's API, from MAIL_FROM, replies to MAIL_REPLY_TO."""
    for k in ("MAIL_TEST_OUTBOX", "RESEND_API_KEY", "SMTP_HOST"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("BREVO_API_KEY", "xkeysib-test")
    monkeypatch.setenv("MAIL_FROM", "GeoEstate <no-reply@geoestate.ng>")
    monkeypatch.setenv("MAIL_REPLY_TO", "support@geoestate.ng")
    sent = []

    class R:
        status_code = 201
        text = "{}"

    monkeypatch.setattr(mailer.httpx, "post", lambda url, json, headers, timeout: sent.append((url, json, headers)) or R())
    assert mailer.backend() == "brevo" and mailer.configured()
    mailer.send("ada@example.com", "FarmScan: 1 farm to check", "Dry spell")
    url, body, headers = sent[0]
    assert url == "https://api.brevo.com/v3/smtp/email" and headers["api-key"] == "xkeysib-test"
    assert body["sender"] == {"name": "GeoEstate", "email": "no-reply@geoestate.ng"} and body["to"] == [{"email": "ada@example.com"}]
    assert body["replyTo"] == {"email": "support@geoestate.ng"} and body["textContent"] == "Dry spell"


def test_fires_come_from_noaa20_and_noaa21(monkeypatch):
    """Suomi NPP stopped on 2 Nov 2026: fires are read from NOAA-20 and NOAA-21; one failing source does not stop the other."""
    monkeypatch.setenv("FIRMS_MAP_KEY", "k")
    asked = []

    class R:
        def __init__(self, text): self.text = text
        def raise_for_status(self): pass

    def get(url, timeout):
        asked.append(url)
        if "NOAA21" in url:
            return R("Invalid MAP_KEY.")
        return R("latitude,longitude,acq_date\n9.072,7.402,2026-11-05\n")
    import requests
    monkeypatch.setattr(requests, "get", get)
    got = alerts.fetch_fires(POLY)
    assert len(got) == 1 and any("VIIRS_NOAA20_NRT" in u for u in asked) and any("VIIRS_NOAA21_NRT" in u for u in asked)
    assert not any("SNPP" in u for u in asked)
    monkeypatch.setattr(requests, "get", lambda url, timeout: R("Invalid MAP_KEY."))
    with pytest.raises(RuntimeError):
        alerts.fetch_fires(POLY)


def test_rainy_months_from_the_real_climate_summary():
    """weather.climate_summary names months ("Apr"); the alerts read them as numbers (this broke /api/alerts)."""
    hist = {"dates": [], "rain": [], "tmax": [], "tmin": [], "source": "test"}
    d = date(2016, 1, 1)
    while d.year < 2026:
        hist["dates"].append(d.isoformat()); hist["rain"].append(4.0 if 4 <= d.month <= 10 else 0.2)
        hist["tmax"].append(32.0); hist["tmin"].append(22.0)
        d += timedelta(days=1)
    real = weather.climate_summary(hist)
    assert alerts.wet_months_of(real) == list(range(4, 11))
    assert alerts.wet_months_of({"wet_months": [5, 6]}) == [5, 6] and alerts.wet_months_of({}) is None


def test_dry_down_after_the_peak_is_not_a_crop_alert():
    trend = {"alerts": [{"obs_date": "2026-10-10", "message": "down 0.12 since the previous image"}]}
    assert alerts.crop_alerts(trend, "ndvi", date(2026, 10, 20)) != []
    assert alerts.crop_alerts(trend, "ndvi", date(2026, 10, 20), after_peak="2026-08-25") == []          # drying down
    assert alerts.crop_alerts(trend, "ndvi", date(2026, 10, 20), after_peak="2026-10-15") != []          # before the peak
    assert alerts.crop_alerts(trend, "ndvi", date(2026, 10, 5)) == []                                    # not yet happened


def test_farm_report_pdf_renders_every_section():
    """Every section of the merged report draws, including an uncertain season, a farm with no season, alerts,
    suitability and a land register that did not answer; and a report issued before the merge still opens."""
    import reports
    sq = {"type": "Polygon", "coordinates": [[[7.4, 9.07], [7.41, 9.07], [7.41, 9.08], [7.4, 9.08], [7.4, 9.07]]]}
    base = {"code": "FSV-TEST0001", "issued_on": "2026-11-07", "org_name": "Kuje Growers", "verify_url": "https://x/verify.html?code=FSV-TEST0001",
            "farm": {"name": "Big farm", "area_ha": 120.5, "lat": 9.075, "lng": 7.405, "declared_ha": 100, "declared_vs_measured_pct": -17,
                     "state": "Kaduna", "farmer": "A. Musa", "crops": ["maize"]},
            "geometry": sq, "images": {"optical": 12, "radar": 20},
            "seasons": [{"year": 2025, "based_on": "rvi", "confidence": "low", "planting": {"from": "2025-05-01", "to": "2025-05-22"},
                         "greenup": {"date": "2025-05-29", "plus_minus_days": 6}, "peak": {"date": "2025-08-10", "plus_minus_days": None},
                         "end": None, "declared_check": {"words": "Green-up came long after the sowing date given", "declared": "2025-03-01"}},
                        {"year": 2026, "based_on": None, "why": "Tree cover, bush or a perennial crop: green all year."}],
            "rainfall": {"years": [{"year": 2025, "rain_mm": 1100, "vs_normal_pct": -8}], "normal_mm": 1200, "normal_years": "2016–2025",
                         "wet_months": ["May", "June"], "note": "A little dry."}}
    full = dict(base, kind="verification", health=None,
                alerts={"alerts": [{"words": "Dry spell: 12 days without rain in the rainy season."}], "unavailable": ["fire"]},
                suitability={"texture": "loam", "ph": 6.1, "crops": [{"label": "Maize", "class": "S2", "class_label": "moderately suitable",
                                                                       "index": 68, "limit": "rain", "window": "May"}]},
                title={"ok": False, "reason": "Land Registry did not answer: LAND_API should end in /api."})
    for c in (full, dict(base, kind="farm", title="not_in_plan"), dict(base, title=None)):   # the last is an old report
        pdf = reports.build_pdf(c)
        assert pdf[:4] == b"%PDF" and len(pdf) > 2500
