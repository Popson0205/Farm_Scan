"""Who is signed in, through GeoEstate ID (Geohub). FarmScan has no accounts of its own.

The access token (15 minutes) and refresh token (30 days) are kept in http-only cookies; when the access token runs
out, the next request refreshes both, which also checks with Geohub that the organization still has FarmScan.

Roles come from the person's role in their Geohub organization:
  manager    = owner or admin: everything, including deleting farms
  agronomist = member: adds farms, runs imagery, plots, zones, notes
  viewer     = collector or viewer, and GeoEstate staff: reads only
The plan's features (monitoring, zones, suitability, drone, plant_count, prescriptions) and its hectare limit come
from the licence. An expired licence is read-only and a suspended one closed, as in Geohub."""
import secrets

from fastapi import Depends, HTTPException, Request, Response

from config import settings
from geoestate_id import GeoEstateID, GeoEstateIDError  # noqa: F401  (re-exported for tests)

AT, RT = "farm_at", "farm_rt"
ROLE_OF = {"owner": "manager", "admin": "manager", "member": "agronomist", "collector": "viewer", "viewer": "viewer"}
RANK = {"viewer": 0, "agronomist": 1, "manager": 2}
FEATURE_WORDS = {"monitoring": "Satellite monitoring", "zones": "Management zones", "suitability": "Crop suitability",
                 "drone": "Drone imagery", "plant_count": "Plant counts", "prescriptions": "Variable-rate prescriptions"}
gid: GeoEstateID | None = None       # tests swap this for one that talks to a stand-in Geohub


class Problem(HTTPException):
    """An error the person can act on: {"detail": words, "code": machine-readable}."""
    def __init__(self, status: int, detail: str, code: str = "error"):
        super().__init__(status, detail)
        self.code = code


def client() -> GeoEstateID:
    global gid
    if gid is None:
        gid = GeoEstateID(geohub=settings.geohub_url, api=settings.geohub_api, client_id=settings.id_client,
                          client_secret=settings.id_secret or None)
    return gid


def callback_url() -> str:
    return f"{settings.app_url}/api/auth/callback"


def keep(response: Response, tokens: dict) -> None:
    opts = {"httponly": True, "secure": settings.cookie_secure, "samesite": "lax", "path": "/"}
    response.set_cookie(AT, tokens["access_token"], max_age=tokens["expires_in"], **opts)
    response.set_cookie(RT, tokens["refresh_token"], max_age=30 * 86400, **opts)


def forget(response: Response) -> None:
    for k in (AT, RT):
        response.delete_cookie(k, path="/")


def new_state() -> tuple[str, str]:
    return secrets.token_urlsafe(48), secrets.token_urlsafe(24)


def person(claims: dict) -> dict:
    p = claims.get("product") or {}
    org = claims.get("org") or {}
    role = "viewer" if claims.get("staff_role") and not p else ROLE_OF.get(org.get("role", ""), "viewer")
    status = p.get("status")
    return {"id": claims["sub"], "name": claims["name"], "email": claims["email"], "org_id": org.get("id"),
            "org_name": org.get("name"), "org_role": org.get("role"), "role": role, "staff_role": claims.get("staff_role"),
            "plan": p.get("plan"), "licence": status, "features": list(p.get("features", [])), "limits": p.get("limits", {}),
            "ends_on": p.get("ends_on"), "can_write": status in ("demo", "active", "grace")}


def current_user(request: Request, response: Response) -> dict:
    at, rt = request.cookies.get(AT), request.cookies.get(RT)
    g = client()
    claims = None
    if at:
        try:
            claims = g.verify(at)
        except GeoEstateIDError:
            claims = None
    if claims is None and rt:
        try:
            tokens = g.refresh(rt)
        except GeoEstateIDError as e:
            forget(response)
            raise Problem(401, str(e) if e.code == "no_product" else "Your sign-in has ended. Sign in again.",
                          code=e.code if e.code == "no_product" else "unauthenticated")
        keep(response, tokens)
        at = tokens["access_token"]
        claims = g.verify(at)
    if claims is None:
        raise Problem(401, "Sign in to continue.", code="unauthenticated")
    u = person(claims)
    if not u["org_id"]:
        raise Problem(403, "Open FarmScan from your organization's account.", code="no_org")
    if u["licence"] == "suspended":
        raise Problem(403, f"{u['org_name']}'s FarmScan access is suspended. Contact GeoEstate NIG Limited to restore it.",
                      code="licence_suspended")
    request.state.at = at
    return u


def need(role: str = "viewer", write: bool = False, feature: str | None = None):
    """A signed-in person with at least this role; `write` also needs a licence that allows changes; `feature` needs
    the organization's plan to include it."""
    def dep(user=Depends(current_user)) -> dict:
        if RANK[user["role"]] < RANK[role]:
            words = {"agronomist": "Only managers and agronomists can do this.", "manager": "Only managers can do this."}
            raise Problem(403, words.get(role, "You cannot do this."), code="forbidden")
        if write and not user["can_write"]:
            raise Problem(403, f"{user['org_name']}'s FarmScan subscription has ended, so it is read-only: you can view and "
                               "export but not add or run anything. Renew with GeoEstate NIG Limited.", code="licence_expired")
        if feature and feature not in user["features"]:
            raise Problem(403, f"{FEATURE_WORDS.get(feature, feature)} is not in {user['org_name']}'s FarmScan plan. "
                               "Ask GeoEstate NIG Limited to add it.", code="feature_not_in_plan")
        return user
    return dep


reader = need()
writer = need("agronomist", write=True)
