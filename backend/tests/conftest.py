"""Tests run against PostgreSQL with PostGIS (DATABASE_URL, default the farm_test database) as one signed-in person
of a test organization. `as_user(...)` changes who that is for a test; `signed_out()` removes them."""
import os
import sys
import uuid
from pathlib import Path

os.environ.setdefault("DATABASE_URL", "postgresql://hub:hub@localhost:5432/farm_test")
os.environ["DEMO_TOOLS"] = "true"
os.environ.setdefault("APP_URL", "http://farm.test")
os.environ.setdefault("GEOHUB_URL", "http://geohub.test")
os.environ.setdefault("GEOHUB_API", "http://geohub.test/api")
for k in ("S3_ENDPOINT", "S3_BUCKET", "S3_ACCESS_KEY"):
    os.environ.pop(k, None)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

import db  # noqa: E402
import identity  # noqa: E402

ORG = "11111111-1111-4111-8111-111111111111"
ALL = ["monitoring", "zones", "suitability", "alerts", "prescriptions", "verification"]


def make_user(**kw) -> dict:
    u = {"id": "u1", "name": "Ada Farmer", "email": "ada@example.com", "org_id": ORG, "org_name": "Test Farms Ltd",
         "org_role": "owner", "role": "manager", "staff_role": None, "plan": "advanced", "licence": "active",
         "features": list(ALL), "limits": {"hectares": 5000}, "ends_on": None, "can_write": True}
    return u | kw


CURRENT: dict = {"user": make_user()}


def _current():
    if CURRENT["user"] is None:
        raise identity.Problem(401, "Sign in to continue.", code="unauthenticated")
    return CURRENT["user"]


@pytest.fixture(scope="session", autouse=True)
def database():
    db.start()
    with db.tx() as conn:
        conn.execute("TRUNCATE boundaries, api_cache CASCADE")
    import main
    main.app.dependency_overrides[identity.current_user] = _current
    yield
    main.app.dependency_overrides.clear()
    db.pool.close()


@pytest.fixture(autouse=True)
def default_user():
    CURRENT["user"] = make_user()
    yield
    CURRENT["user"] = make_user()


@pytest.fixture
def as_user():
    def set_user(**kw):
        CURRENT["user"] = make_user(**kw)
        return CURRENT["user"]
    return set_user


@pytest.fixture
def other_org():
    return str(uuid.uuid4())


@pytest.fixture(autouse=True)
def no_real_soilgrids(monkeypatch):
    """Tests never reach the real SoilGrids (its answer would be cached and read by later tests at the same place);
    tests that need it fake requests.get themselves."""
    import requests
    import soilgrids
    real = requests.get

    def get(url, *a, **k):
        if str(url).startswith(soilgrids.URL):
            raise requests.ConnectionError("no network in tests")
        return real(url, *a, **k)
    monkeypatch.setattr(requests, "get", get)
