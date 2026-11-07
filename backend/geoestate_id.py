"""Sign in with GeoEstate ID, for a GeoEstate app with a Python server (FastAPI or any other).

Copy this one file into the app. It needs `PyJWT[crypto]` and `httpx`.

    from geoestate_id import GeoEstateID
    gid = GeoEstateID(geohub="https://hub.geoestate.ng", api="https://hub.geoestate.ng/api",
                      client_id="farm", client_secret=os.environ["GEOESTATE_ID_SECRET"])

    # 1. "Sign in": send the browser to Geohub
    verifier, state = gid.new_verifier(), gid.new_state()        # keep both (a cookie or the session) until the return
    return RedirectResponse(gid.authorize_url(redirect_uri, state, verifier))

    # 2. Back at redirect_uri?code=…&state=…: check state, then swap the code
    tokens = gid.exchange(code, redirect_uri, verifier)            # {"access_token", "refresh_token", "expires_in", "user"}

    # 3. On every request: who is it, and what does their licence allow?
    me = gid.verify(access_token)                                   # raises GeoEstateIDError when not valid
    me["org"]["role"], me["product"]["status"], "dxf_import" in me["product"]["features"]

    # 4. Every 15 minutes: gid.refresh(refresh_token) — which also checks they still have the product.

The access token is checked with Geohub's public key, fetched once and cached; no call to Geohub per request."""
import base64
import hashlib
import secrets
import time
from urllib.parse import urlencode

import httpx
import jwt


class GeoEstateIDError(Exception):
    def __init__(self, message: str, status: int = 401, code: str = "invalid_token"):
        super().__init__(message)
        self.status, self.code = status, code


class GeoEstateID:
    def __init__(self, geohub: str, api: str, client_id: str, client_secret: str | None = None,
                 transport: httpx.BaseTransport | None = None, key_ttl: int = 3600):
        self.geohub, self.api = geohub.rstrip("/"), api.rstrip("/")
        self.client_id, self.client_secret = client_id, client_secret
        self._http = httpx.Client(timeout=15, transport=transport)
        self._keys: dict[str, object] = {}
        self._keys_at = 0.0
        self.key_ttl = key_ttl

    # ---- signing in ----
    @staticmethod
    def new_verifier() -> str:
        return secrets.token_urlsafe(48)

    @staticmethod
    def new_state() -> str:
        return secrets.token_urlsafe(24)

    @staticmethod
    def challenge(verifier: str) -> str:
        return base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()

    def authorize_url(self, redirect_uri: str, state: str, verifier: str) -> str:
        q = {"client_id": self.client_id, "redirect_uri": redirect_uri, "state": state,
             "code_challenge": self.challenge(verifier), "code_challenge_method": "S256"}
        return f"{self.geohub}/id/authorize?{urlencode(q)}"

    def _post(self, path: str, data: dict) -> dict:
        body = {"client_id": self.client_id, **({"client_secret": self.client_secret} if self.client_secret else {}), **data}
        r = self._http.post(f"{self.api}/id/{path}", data=body)
        if r.status_code >= 400:
            try:
                p = r.json()
            except ValueError:
                p = {}
            raise GeoEstateIDError(p.get("detail") or f"Geohub answered {r.status_code}", r.status_code, p.get("code", "error"))
        return r.json()

    def exchange(self, code: str, redirect_uri: str, verifier: str) -> dict:
        return self._post("token", {"grant_type": "authorization_code", "code": code, "redirect_uri": redirect_uri,
                                    "code_verifier": verifier})

    def refresh(self, refresh_token: str) -> dict:
        return self._post("token", {"grant_type": "refresh_token", "refresh_token": refresh_token})

    def sign_out(self, refresh_token: str) -> None:
        self._post("revoke", {"token": refresh_token})

    def apps(self, access_token: str) -> list[dict]:
        """The GeoEstate apps the person's organization has, for an app switcher: product, name, status, url, current."""
        r = self._http.get(f"{self.api}/id/apps", headers={"Authorization": f"Bearer {access_token}"})
        if r.status_code >= 400:
            raise GeoEstateIDError(r.json().get("detail", "Geohub refused"), r.status_code)
        return r.json()

    def data(self, access_token: str, path: str, params: dict | None = None):
        """Geohub's field data, read as the person (GET /id/data/...): "projects", "forms/{id}/features", "features/{id}".
        Only what they could open in Geohub themselves."""
        r = self._http.get(f"{self.api}/id/data/{path.lstrip('/')}", params=params or {}, headers={"Authorization": f"Bearer {access_token}"})
        if r.status_code >= 400:
            try:
                body = r.json()
            except ValueError:
                body = {}
            raise GeoEstateIDError(body.get("detail", f"Geohub answered {r.status_code}"), r.status_code, body.get("code") or "error")
        return r.json()

    # ---- checking an access token ----
    def _key(self, kid: str | None):
        if kid not in self._keys or time.time() - self._keys_at > self.key_ttl:
            r = self._http.get(f"{self.api}/id/jwks")
            r.raise_for_status()
            self._keys = {k["kid"]: jwt.PyJWK(k).key for k in r.json()["keys"]}
            self._keys_at = time.time()
        if kid not in self._keys:
            raise GeoEstateIDError("The token was signed with a key Geohub does not publish.")
        return self._keys[kid]

    def verify(self, access_token: str) -> dict:
        """The person's claims: sub, email, name, org {id, name, role}, product {id, plan, status, features, limits…},
        staff_role. Raises GeoEstateIDError when the token is not for this app, expired, or not Geohub's."""
        try:
            kid = jwt.get_unverified_header(access_token).get("kid")
            claims = jwt.decode(access_token, self._key(kid), algorithms=["EdDSA"], audience=self.client_id, issuer=self.geohub)
        except jwt.ExpiredSignatureError:
            raise GeoEstateIDError("The sign-in has expired.", code="token_expired")
        except jwt.InvalidTokenError as e:
            raise GeoEstateIDError(f"That token is not valid ({e}).")
        if claims.get("typ") != "geoestate-id":
            raise GeoEstateIDError("That token is not a GeoEstate ID access token.")
        return claims

    @staticmethod
    def can_write(claims: dict) -> bool:
        """Expired licences are read-only, suspended ones closed: the same rule as Geohub's own."""
        p = claims.get("product")
        return bool(p) and p["status"] in ("demo", "active", "grace")
