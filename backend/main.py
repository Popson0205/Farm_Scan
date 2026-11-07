import base64
import io
import json
import logging
import tempfile
import uuid
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
import threading
from datetime import date as date_cls, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
from fastapi import Depends, FastAPI, HTTPException, Request, Response, UploadFile, File, Form
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from psycopg.types.json import Jsonb
from shapely.geometry import shape, mapping

from indices import INDEX_REGISTRY, compute_index, zonal_stats
import db
import demo_data
import geo
import identity
import storage
import zones as zones_mod
import exports as exports_mod
import annotations as annotations_mod
import monitoring as monitoring_mod
import season as season_mod
import alerts as alerts_mod
import mailer
import reports as reports_mod
import soil as soil_mod
import weather as weather_mod
import suitability as suitability_mod
from config import settings
from geoestate_id import GeoEstateIDError
from identity import Problem, current_user, need, reader, writer

log = logging.getLogger("farmscan")

# In-memory cache of the last computed index result per (boundary_id, index), so plot stats / zones / exports can
# reuse the raster instead of recomputing. Farm ids are checked against the person's organization before use.
_RESULT_CACHE: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(_app):
    db.start()
    with db.tx() as conn:                            # jobs cut short by a restart
        conn.execute("UPDATE season_jobs SET status = 'failed', message = 'FarmScan restarted while this ran: start it again.', "
                     "finished_at = now() WHERE status IN ('queued', 'running')")
    soil_mod.fetch_from_bucket()
    stop = threading.Event()
    threading.Thread(target=_alerts_loop, args=(stop,), daemon=True, name="weekly-alerts").start()
    yield
    stop.set()
    db.pool.close()


app = FastAPI(title="FarmScan API", description="Satellite farm health mapping, on the GeoEstate platform",
              lifespan=lifespan)


@app.exception_handler(HTTPException)
async def _problem(request: Request, exc: HTTPException):
    body = {"detail": exc.detail}
    if isinstance(exc, Problem):
        body["code"] = exc.code
    return JSONResponse(body, status_code=exc.status_code, headers=getattr(exc, "headers", None))


def _uuid(v: str, what: str = "Farm") -> str:
    try:
        return str(uuid.UUID(str(v)))
    except (ValueError, TypeError):
        raise Problem(404, f"{what} not found", code="not_found")


def _farm(user: dict, boundary_id: str, conn=None) -> dict:
    """The farm, if it belongs to the person's organization (404 otherwise, as if it did not exist)."""
    bid = _uuid(boundary_id)
    sql = ("SELECT id::text AS id, name, geojson, area_ha, created_by, created_at::text AS created_at, details, source FROM boundaries "
           "WHERE id = %s AND org_id = %s")
    if conn is not None:
        row = conn.execute(sql, (bid, user["org_id"])).fetchone()
    else:
        with db.tx() as c:
            row = c.execute(sql, (bid, user["org_id"])).fetchone()
    if not row:
        raise Problem(404, "Farm not found", code="not_found")
    return row


def _geometry(parsed: dict) -> dict:
    """A bare Polygon/MultiPolygon geometry from a Feature, FeatureCollection or geometry."""
    if parsed.get("type") == "FeatureCollection":
        if not parsed.get("features"):
            raise HTTPException(400, "The FeatureCollection has no features")
        parsed = parsed["features"][0]["geometry"]
    elif parsed.get("type") == "Feature":
        parsed = parsed["geometry"]
    if not parsed or parsed.get("type") not in ("Polygon", "MultiPolygon"):
        raise HTTPException(400, "A farm boundary must be a Polygon or MultiPolygon")
    return parsed


def _valid_shape(geometry: dict):
    try:
        g = shape(geometry)
        if not g.is_valid or g.is_empty:
            raise ValueError("Geometry is invalid or empty")
        return g
    except Exception as e:
        raise HTTPException(400, f"Could not parse geometry: {e}")


# ---------------------------------------------------------------------------
# Signing in: GeoEstate ID (Geohub)
# ---------------------------------------------------------------------------

def _safe(path: str | None) -> str:
    return path if path and path.startswith("/") and not path.startswith("//") else "/"


@app.get("/api/auth/sign-in", summary="Off to Geohub to sign in; back to `next` afterwards")
def sign_in(next: str | None = None):
    verifier, state = identity.new_state()
    r = RedirectResponse(identity.client().authorize_url(identity.callback_url(), state, verifier), status_code=303)
    opts = {"httponly": True, "secure": settings.cookie_secure, "samesite": "lax", "max_age": 600, "path": "/"}
    r.set_cookie("farm_pkce", verifier, **opts)
    r.set_cookie("farm_state", state, **opts)
    r.set_cookie("farm_next", _safe(next), **opts)
    return r


@app.get("/api/auth/callback", summary="Back from Geohub with a one-time code")
def auth_callback(request: Request, code: str = "", state: str = ""):
    if not code or not state or state != request.cookies.get("farm_state"):
        return RedirectResponse(f"{settings.app_url}/?signin_error={quote('This sign-in did not start here. Sign in again.')}", 303)
    try:
        tokens = identity.client().exchange(code, identity.callback_url(), request.cookies.get("farm_pkce", ""))
    except GeoEstateIDError as e:
        return RedirectResponse(f"{settings.app_url}/?signin_error={quote(str(e))}", 303)
    r = RedirectResponse(f"{settings.app_url}{_safe(request.cookies.get('farm_next'))}", 303)
    identity.keep(r, tokens)
    for k in ("farm_pkce", "farm_state", "farm_next"):
        r.delete_cookie(k, path="/")
    return r


@app.post("/api/auth/sign-out", status_code=204, summary="Sign out of FarmScan (Geohub stays signed in)")
def sign_out(request: Request):
    r = Response(status_code=204)
    rt = request.cookies.get(identity.RT)
    if rt:
        try:
            identity.client().sign_out(rt)
        except GeoEstateIDError:
            pass
    identity.forget(r)
    return r


@app.get("/api/me", summary="Who is signed in, their role, the plan's features and hectares used")
def me(user=Depends(current_user)):
    _remember_licence(user)
    with db.tx() as conn:
        used = conn.execute("SELECT coalesce(sum(area_ha), 0) AS ha, count(*) AS farms FROM boundaries WHERE org_id = %s",
                            (user["org_id"],)).fetchone()
    return user | {"hectares_used": round(used["ha"], 2), "farms": used["farms"],
                   "hectares_limit": (user["limits"] or {}).get("hectares"), "geohub_url": settings.geohub_url,
                   "demo_tools": settings.demo_tools}


@app.get("/api/apps", summary="The GeoEstate apps the organization has, for the app switcher")
def apps(request: Request, user=Depends(current_user)):
    token = getattr(request.state, "at", None) or request.cookies.get(identity.AT)
    try:
        return identity.client().apps(token) if token else []
    except GeoEstateIDError:
        return []


# ---------------------------------------------------------------------------
# Farms (boundaries)
# ---------------------------------------------------------------------------

@app.post("/api/boundary")
def create_boundary(name: str = Form(...), geojson: str = Form(...), user=Depends(writer)):
    """Add a farm/field boundary. `geojson` is a stringified GeoJSON Feature, FeatureCollection, or bare
    Polygon/MultiPolygon geometry (EPSG:4326). Counts against the plan's hectares."""
    try:
        parsed = json.loads(geojson)
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid JSON in geojson field")
    if not isinstance(parsed, dict):
        raise HTTPException(400, "geojson must be a GeoJSON object")
    geometry = _geometry(parsed)
    geom = _valid_shape(geometry)
    area_ha = round(geo.area_m2(geom) / 10000, 2)
    if area_ha <= 0:
        raise HTTPException(400, "The boundary has no area")
    name = name.strip()[:200] or "Unnamed farm"
    boundary_id = str(uuid.uuid4())
    with db.tx() as conn:
        conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (user["org_id"],))     # one farm added at a time per org
        limit = (user["limits"] or {}).get("hectares")
        if limit is not None:
            used = conn.execute("SELECT coalesce(sum(area_ha), 0) AS ha FROM boundaries WHERE org_id = %s",
                                (user["org_id"],)).fetchone()["ha"]
            if used + area_ha > limit:
                raise Problem(403, f"This farm ({area_ha:,.2f} ha) would take {user['org_name']} to {used + area_ha:,.2f} ha, over "
                                   f"its plan's {limit:,} ha. Remove a farm you no longer monitor, or ask GeoEstate NIG Limited "
                                   "for more hectares.", code="hectares_limit")
        conn.execute("INSERT INTO boundaries (id, org_id, name, geojson, area_ha, created_by) VALUES (%s, %s, %s, %s, %s, %s)",
                     (boundary_id, user["org_id"], name, Jsonb(geometry), area_ha, user["name"]))
    return {"boundary_id": boundary_id, "name": name, "area_ha": area_ha,
            "centroid": [geom.centroid.y, geom.centroid.x], "bounds": list(geom.bounds)}


@app.get("/api/boundary/{boundary_id}")
def get_boundary(boundary_id: str, user=Depends(reader)):
    row = _farm(user, boundary_id)
    return {"boundary_id": row["id"], "name": row["name"], "geojson": row["geojson"], "area_ha": row["area_ha"],
            "created_by": row["created_by"], "created_at": row["created_at"], "details": row["details"], "source": row["source"]}


@app.get("/api/boundaries")
def list_boundaries(user=Depends(reader)):
    with db.tx() as conn:
        return conn.execute("SELECT id::text AS id, name, area_ha, created_by, created_at::text AS created_at, source, "
                            "details->>'farmer_name' AS farmer FROM boundaries "
                            "WHERE org_id = %s ORDER BY created_at DESC", (user["org_id"],)).fetchall()


@app.delete("/api/boundary/{boundary_id}", status_code=204)
def delete_boundary(boundary_id: str, user=Depends(need("manager", write=True))):
    """Remove a farm with its plots, timeline, notes and share links (frees its hectares)."""
    with db.tx() as conn:
        _farm(user, boundary_id, conn)
        conn.execute("DELETE FROM boundaries WHERE id = %s AND org_id = %s", (_uuid(boundary_id), user["org_id"]))
    for k in [k for k in _RESULT_CACHE if k.startswith(f"{_uuid(boundary_id)}:")]:
        _RESULT_CACHE.pop(k, None)
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# Available indices
# ---------------------------------------------------------------------------

@app.get("/api/indices")
async def list_indices():
    return {
        name: {"label": spec["label"], "bands_required": spec["bands"], "typical_range": spec["range"]}
        for name, spec in INDEX_REGISTRY.items()
    }


# ---------------------------------------------------------------------------
# Scene search (live satellite catalog)
# ---------------------------------------------------------------------------

@app.get("/api/imagery/search")
def search_imagery(boundary_id: str, start_date: str, end_date: str, max_cloud_cover: float = 30.0,
                   max_scenes: int = 10, user=Depends(need(feature="monitoring"))):
    """Search Sentinel-2 scenes. Returns only the best `max_scenes` (default 10) that cover the
    whole boundary, clearest first. Requires outbound network access to
    planetarycomputer.microsoft.com — falls back with a clear error if unreachable."""
    geometry = _farm(user, boundary_id)["geojson"]
    try:
        import stac_client
        found = stac_client.search_scenes_detailed(geometry, start_date, end_date, max_cloud_cover,
                                                   limit=max(1, min(int(max_scenes), 50)))
        scenes = found["scenes"]
        # strip the href cache before sending to client; keep it server-side keyed by scene_id
        _SCENE_CACHE.update({s["scene_id"]: s["_href_cache"] for s in scenes})
        _SCENE_META.update({s["scene_id"]: {"date": (s.get("date") or "")[:10],
                                            "cloud_cover": s.get("cloud_cover"),
                                            "offset": s.get("_reflectance_offset", 0.0)}
                            for s in scenes})
        for s in scenes:
            s.pop("_href_cache", None)
            s.pop("_reflectance_offset", None)
        return {"scenes": scenes, "source": "live",
                "checked": found["checked"], "fully_covering": found["fully_covering"],
                "max_scenes": max(1, min(int(max_scenes), 50))}
    except Exception as e:
        raise HTTPException(
            502,
            f"Could not reach the satellite catalogue ({e}). Try again in a few minutes."
        )


_SCENE_CACHE: dict[str, dict] = {}  # scene_id -> band href map, populated by search_imagery
_SCENE_META: dict[str, dict] = {}   # scene_id -> {date, cloud_cover, offset}


# ---------------------------------------------------------------------------
# Index processing (live)
# ---------------------------------------------------------------------------

@app.post("/api/imagery/process")
def process_imagery(boundary_id: str = Form(...), scene_id: str = Form(...), index: str = Form(...),
                    user=Depends(need("agronomist", write=True, feature="monitoring"))):
    if index not in INDEX_REGISTRY:
        raise HTTPException(400, f"Unknown index '{index}'. Choose from {list(INDEX_REGISTRY)}")

    href_map = _SCENE_CACHE.get(scene_id)
    if not href_map:
        raise HTTPException(400, "Unknown scene — search for scenes again first")
    farm = _farm(user, boundary_id)
    boundary_id, boundary_geojson = farm["id"], farm["geojson"]

    import stac_client
    bands_needed = INDEX_REGISTRY[index]["bands"]
    meta = _SCENE_META.get(scene_id, {})
    # SCL cloud/shadow masking + common grid + reflectance offset are handled in stac_client
    band_arrays, out_meta = stac_client.fetch_and_clip_bands(
        href_map, boundary_geojson, bands_needed, reflectance_offset=meta.get("offset", 0.0))

    return _finish_processing(index, band_arrays, out_meta, boundary_geojson, boundary_id=boundary_id,
                              obs_date=meta.get("date") or date_cls.today().isoformat(),
                              source="live", scene_id=scene_id,
                              cloud_cover=meta.get("cloud_cover"),
                              valid_fraction=out_meta.get("valid_fraction"))


# ---------------------------------------------------------------------------
# Index processing (demo / offline mode — synthetic data, no network needed)
# ---------------------------------------------------------------------------

@app.post("/api/imagery/process-demo")
def process_imagery_demo(boundary_id: str = Form(...), index: str = Form(...), obs_date: str = Form(None),
                         user=Depends(writer)):
    """Offline synthetic run. Optionally pass obs_date (YYYY-MM-DD) to record the result as
    that date; each date seeds different synthetic data, which lets you exercise the
    season timeline / alerts without live imagery. Only where DEMO_TOOLS is on (sales demonstrations): sample results must never be taken
    for a customer's farm, and they are kept apart from real images in the timeline (source "demo")."""
    if not settings.demo_tools:
        raise Problem(404, "Sample data is not available here.", code="not_found")
    if index not in INDEX_REGISTRY:
        raise HTTPException(400, f"Unknown index '{index}'. Choose from {list(INDEX_REGISTRY)}")
    farm = _farm(user, boundary_id)
    boundary_id, boundary_geojson = farm["id"], farm["geojson"]

    seed = 0
    if obs_date:
        try:
            seed = date_cls.fromisoformat(obs_date).toordinal()
        except ValueError:
            raise HTTPException(400, "obs_date must be YYYY-MM-DD")
    band_arrays, out_meta, field_mask = demo_data.synthetic_bands_for_boundary(boundary_geojson, seed=seed)
    return _finish_processing(index, band_arrays, out_meta, boundary_geojson, field_mask, demo=True,
                               boundary_id=boundary_id,
                               obs_date=obs_date or date_cls.today().isoformat(),
                               source="demo", valid_fraction=1.0)


def _colorize(index_array: np.ndarray, colormap: str, value_range: tuple) -> Image.Image:
    """Map a float index array to an RGBA image using a simple 3-stop gradient
    (avoids a matplotlib dependency)."""
    vmin, vmax = value_range
    norm = np.clip((index_array - vmin) / (vmax - vmin), 0, 1)

    gradients = {
        "RdYlGn": [(215, 48, 39), (255, 255, 191), (26, 152, 80)],
        "Blues": [(247, 251, 255), (107, 174, 214), (8, 48, 107)],
        "BrBG": [(140, 81, 10), (245, 245, 245), (1, 102, 94)],
    }
    stops = gradients.get(colormap, gradients["RdYlGn"])
    stops = np.array(stops, dtype=np.float32)

    h, w = norm.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    lower_half = norm < 0.5
    t_lo = np.clip(norm / 0.5, 0, 1)
    t_hi = np.clip((norm - 0.5) / 0.5, 0, 1)

    for c in range(3):
        rgb[..., c] = np.where(
            lower_half,
            stops[0, c] + (stops[1, c] - stops[0, c]) * t_lo,
            stops[1, c] + (stops[2, c] - stops[1, c]) * t_hi,
        )

    alpha = np.where(np.isnan(index_array), 0, 255).astype(np.uint8)
    rgba = np.dstack([rgb.astype(np.uint8), alpha])
    return Image.fromarray(rgba, mode="RGBA")


def _finish_processing(index: str, band_arrays: dict, out_meta: dict,
                        boundary_geojson: dict, field_mask: np.ndarray = None, demo: bool = False,
                        boundary_id: str = None, obs_date: str = None, source: str = "live",
                        scene_id: str = None, cloud_cover: float = None,
                        valid_fraction: float = None):
    idx_array = compute_index(index, band_arrays)

    if field_mask is None:
        field_mask = ~np.isnan(idx_array)

    spec = INDEX_REGISTRY[index]
    img = _colorize(idx_array, spec["colormap"], spec["range"])

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()
    png_b64 = base64.b64encode(png_bytes).decode()

    geom = shape(boundary_geojson)
    minx, miny, maxx, maxy = geom.bounds

    stats = zonal_stats(idx_array, field_mask)

    # cache the raw result so /plots, /zones, /export endpoints can reuse it
    if boundary_id:
        _RESULT_CACHE[f"{boundary_id}:{index}"] = {
            "array": idx_array,
            "transform": out_meta["transform"],
            "crs": out_meta["crs"],
            "png_bytes": png_bytes,
            "stats": stats,
            "label": spec["label"],
            "obs_date": obs_date, "source": source,
        }

    # persist the observation so the field builds a season timeline
    observation = None
    warning = None
    if boundary_id and obs_date and stats["pixel_count"] > 0:
        observation = monitoring_mod.save_observation(
            boundary_id, index, obs_date, source, idx_array, scene_id=scene_id,
            cloud_cover=cloud_cover, valid_fraction=valid_fraction)
        if valid_fraction is not None and valid_fraction < monitoring_mod.MIN_VALID_FRACTION:
            warning = (f"Only {valid_fraction*100:.0f}% of the field is cloud-free in this image. "
                       f"It is saved but excluded from trend/alert analysis; pick a clearer scene.")

    return JSONResponse({
        "index": index,
        "label": spec["label"],
        "demo_mode": demo,
        "overlay_png_base64": png_b64,
        "overlay_bounds": [[miny, minx], [maxy, maxx]],  # Leaflet [[south, west], [north, east]]
        "value_range": spec["range"],
        "colormap": spec["colormap"],
        "stats": stats,
        "valid_fraction": valid_fraction,
        "observation": observation,
        "warning": warning,
    })


def _cached(boundary_id: str, index: str) -> dict:
    cached = _RESULT_CACHE.get(f"{boundary_id}:{index}")
    if not cached:
        raise HTTPException(400, f"No processed '{index}' result for this farm — run the index first.")
    return cached



# ---------------------------------------------------------------------------
# Management zones & prescription export
# ---------------------------------------------------------------------------

@app.post("/api/zones/classify")
def classify_zones_endpoint(boundary_id: str = Form(...), index: str = Form(...),
                            n_zones: int = Form(3), base_rate: float = Form(None),
                            unit: str = Form("kg/ha"), strategy: str = Form("compensate"),
                            spread: float = Form(0.2), user=Depends(need(feature="zones"))):
    """Clusters the index raster into n management zones (e.g. 'low/medium/high
    vigor') and polygonizes them — the input for a variable-rate prescription file.

    If base_rate is given (the agronomist's flat-rate recommendation, e.g. 90 for kg N/ha),
    also returns a rate plan: a rate, area and product total per zone plus a comparison with
    flat-rate application. strategy='compensate' puts more input on weak zones,
    'capitalize' on strong ones; spread is the max +/- change (0.2 = 20%). A rate plan needs Variable-rate
    prescriptions in the plan."""
    boundary_id = _farm(user, boundary_id)["id"]
    if base_rate is not None and "prescriptions" not in user["features"]:
        raise Problem(403, f"Variable-rate prescriptions are not in {user['org_name']}'s FarmScan plan. Zones are; "
                           "leave the base rate empty, or ask GeoEstate NIG Limited to add prescriptions.",
                      code="feature_not_in_plan")
    cached = _cached(boundary_id, index)

    try:
        zone_array = zones_mod.classify_zones(cached["array"], n_zones=n_zones)
        zone_polygons = zones_mod.zones_to_polygons(zone_array, cached["transform"], cached["crs"], n_zones)
    except ValueError as e:
        raise HTTPException(400, str(e))

    rate_plan = None
    if base_rate is not None:
        px_area = monitoring_mod.pixel_area_m2(cached["transform"], cached["crs"], cached["array"].shape)
        try:
            rate_plan = zones_mod.build_rate_plan(zone_polygons, px_area, base_rate, unit, strategy, spread)
        except ValueError as e:
            raise HTTPException(400, str(e))

    # cache for export
    _RESULT_CACHE[f"{boundary_id}:{index}:zones:{n_zones}"] = {"zones": zone_polygons, "rate_plan": rate_plan}

    return {"index": index, "n_zones": n_zones, "zones": zone_polygons, "rate_plan": rate_plan}


# ---------------------------------------------------------------------------
# Field monitoring: season timeline, alerts, problem areas -> scouting
# ---------------------------------------------------------------------------

@app.get("/api/monitoring/history")
def monitoring_history(boundary_id: str, index: str = "ndvi", drop_threshold: float = 0.05,
                       user=Depends(need(feature="monitoring"))):
    """Season timeline for a field: every processed image with its field mean, change vs the
    previous image and vs the recent baseline, plus alerts for meaningful drops. Images that
    are too cloudy are listed but excluded from the analysis."""
    if index not in INDEX_REGISTRY:
        raise HTTPException(400, f"Unknown index '{index}'")
    boundary_id = _farm(user, boundary_id)["id"]
    obs = monitoring_mod.list_observations(boundary_id, index)
    return {"boundary_id": boundary_id, "index": index,
            **monitoring_mod.analyze_trend(obs, drop_threshold=drop_threshold)}


@app.post("/api/monitoring/problem-areas")
def monitoring_problem_areas(boundary_id: str = Form(...), index: str = Form("ndvi"),
                             sensitivity: float = Form(1.5), min_area_m2: float = Form(400.0),
                             create_scouting_notes: bool = Form(False), user=Depends(need(feature="monitoring"))):
    """Finds patches clearly below the field's norm in the last processed image, ranked by
    impact (area x shortfall). With create_scouting_notes=true, drops a note at each patch so a
    scout knows where to walk first (replaces this farm's previous auto-generated notes;
    human-written notes are untouched)."""
    boundary_id = _farm(user, boundary_id)["id"]
    if create_scouting_notes and not (identity.RANK[user["role"]] >= 1 and user["can_write"]):
        raise Problem(403, "Only managers and agronomists can add scouting notes, while the subscription is current.",
                      code="forbidden")
    cached = _cached(boundary_id, index)
    try:
        result = monitoring_mod.find_problem_areas(
            cached["array"], cached["transform"], cached["crs"], index,
            sensitivity=sensitivity, min_area_m2=min_area_m2)
    except ValueError as e:
        raise HTTPException(400, str(e))

    notes_created = 0
    if create_scouting_notes:
        obs_date = cached.get("obs_date") or ""
        notes = [{"lat": a["centroid"]["lat"], "lng": a["centroid"]["lng"],
                  "comment": monitoring_mod.scouting_note_text(a, index, obs_date)}
                 for a in result["areas"]]
        notes_created = len(annotations_mod.replace_auto_scouting_notes(boundary_id, notes))
    return {**result, "obs_date": cached.get("obs_date"), "scouting_notes_created": notes_created}


# ---------------------------------------------------------------------------
# Exports: GeoTIFF, Shapefile, Excel, PDF
# ---------------------------------------------------------------------------

@app.get("/api/export/geotiff")
def export_geotiff(boundary_id: str, index: str, user=Depends(reader)):
    boundary_id = _farm(user, boundary_id)["id"]
    cached = _cached(boundary_id, index)
    tiff_bytes = exports_mod.index_array_to_geotiff_bytes(cached["array"], cached["transform"], cached["crs"])
    return StreamingResponse(
        io.BytesIO(tiff_bytes), media_type="image/tiff",
        headers={"Content-Disposition": f'attachment; filename="{index}_{boundary_id[:8]}.tif"'},
    )


@app.get("/api/export/shp/zones")
def export_zones_shp(boundary_id: str, index: str, n_zones: int = 3, user=Depends(need(feature="zones"))):
    boundary_id = _farm(user, boundary_id)["id"]
    cached = _RESULT_CACHE.get(f"{boundary_id}:{index}:zones:{n_zones}")
    if not cached:
        raise HTTPException(400, "No zones cached — call /api/zones/classify first.")
    features = []
    for z in cached["zones"]:
        f = {"geometry": z["geometry"], "zone": z["zone"], "px_count": z["pixel_count"],
             "label": z["suggested_rate"]}
        if "rate" in z:   # agronomist-set rate plan present
            f.update({"rate": z["rate"], "unit": z["unit"], "area_ha": z["area_ha"],
                      "product": z["product_total"]})
        features.append(f)
    zip_bytes = exports_mod.geometries_to_shapefile_zip_bytes(features, layer_name="zones")
    return StreamingResponse(
        io.BytesIO(zip_bytes), media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="zones_{index}_{boundary_id[:8]}.zip"'},
    )


@app.get("/api/export/pdf")
def export_pdf(boundary_id: str, index: str, user=Depends(reader)):
    farm = _farm(user, boundary_id)
    boundary_id, farm_name = farm["id"], farm["name"]
    cached = _cached(boundary_id, index)

    pdf_bytes = exports_mod.build_pdf_report_bytes(
        farm_name, cached["label"], cached["stats"], overlay_png_bytes=cached["png_bytes"]
    )
    return StreamingResponse(
        io.BytesIO(pdf_bytes), media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="report_{index}_{boundary_id[:8]}.pdf"'},
    )


# ---------------------------------------------------------------------------
# Annotations & sharing
# ---------------------------------------------------------------------------

@app.post("/api/annotations")
def create_annotation(boundary_id: str = Form(...), lat: float = Form(...), lng: float = Form(...),
                      comment: str = Form(...), user=Depends(writer)):
    comment = comment.strip()
    if not comment:
        raise HTTPException(400, "Write a note")
    return annotations_mod.add_annotation(_farm(user, boundary_id)["id"], lat, lng, comment[:2000], user["name"])


@app.get("/api/annotations/{boundary_id}")
def get_annotations(boundary_id: str, user=Depends(reader)):
    return {"annotations": annotations_mod.list_annotations(_farm(user, boundary_id)["id"])}


@app.post("/api/share")
def create_share(boundary_id: str = Form(...), index: str = Form(None), user=Depends(writer)):
    token = annotations_mod.create_share_link(_farm(user, boundary_id)["id"], index, user["name"])
    return {"token": token, "share_path": f"/api/share/{token}"}


@app.get("/api/share/{token}")
def resolve_share(token: str):
    """A read-only view of one farm (and result) for anyone with the link: no sign-in."""
    link = annotations_mod.resolve_share_link(token)
    if not link:
        raise HTTPException(404, "Share link not found or expired")
    with db.tx() as conn:
        b = conn.execute("SELECT id::text AS id, name, geojson, area_ha, created_at::text AS created_at FROM boundaries "
                         "WHERE id = %s", (link["boundary_id"],)).fetchone()
    if not b:
        raise HTTPException(404, "Share link not found or expired")
    result = {"boundary": {"boundary_id": b["id"], "name": b["name"], "geojson": b["geojson"], "area_ha": b["area_ha"],
                           "created_at": b["created_at"]}, "index": link["index_name"]}
    if link["index_name"]:
        cached = _RESULT_CACHE.get(f"{link['boundary_id']}:{link['index_name']}")
        if cached:
            result["overlay_png_base64"] = base64.b64encode(cached["png_bytes"]).decode()
            result["stats"] = cached["stats"]
            result["source"] = cached.get("source")
    return result


# ---------------------------------------------------------------------------
# Site suitability: soil (HWSD v2.0, local) + weather (Open-Meteo / NASA POWER)
# These handlers are plain `def` (not async) on purpose: they make blocking HTTP
# and raster calls, so FastAPI runs them in its threadpool instead of stalling
# the event loop.
# ---------------------------------------------------------------------------

def _boundary_geometry(user: dict, boundary_id: str):
    """(geojson geometry, centroid lat, centroid lon) for one of the organization's farms."""
    geometry = _farm(user, boundary_id)["geojson"]
    c = shape(geometry).centroid
    return geometry, c.y, c.x


def _soil_or_http_error(geometry: dict) -> dict:
    try:
        return soil_mod.soil_profile(geometry)
    except soil_mod.SoilDataUnavailable as e:
        raise HTTPException(503, str(e))


def _weather_or_http_error(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except weather_mod.WeatherUnavailable as e:
        raise HTTPException(502, str(e))


@app.get("/api/data-status")
def data_status(user=Depends(reader)):
    """Is the soil dataset installed? Which weather providers are configured?"""
    return {"soil": soil_mod.status(),
            "weather": {"primary": "Open-Meteo (ERA5 + forecast)", "fallback": "NASA POWER",
                        "commercial_key_configured": bool(weather_mod.OPEN_METEO_KEY)}}


@app.get("/api/soil/profile")
def soil_profile(boundary_id: str, user=Depends(need(feature="suitability"))):
    """Soil constituents under the boundary (sand/silt/clay, pH, organic carbon, CEC, salinity,
    drainage, ...) by depth, from HWSD v2.0 (~1 km). Screening-grade: confirm with a soil test."""
    geometry, _, _ = _boundary_geometry(user, boundary_id)
    return {"boundary_id": boundary_id, **_soil_or_http_error(geometry)}


@app.get("/api/weather/climate")
def weather_climate(boundary_id: str, years: int = 10, user=Depends(need(feature="suitability"))):
    """Monthly rainfall/temperature normals + rainy season from the last `years` full years."""
    years = max(1, min(int(years), 30))
    _, lat, lon = _boundary_geometry(user, boundary_id)
    series = _weather_or_http_error(weather_mod.get_daily_history, lat, lon, years)
    return {"boundary_id": boundary_id, "lat": lat, "lon": lon, **weather_mod.climate_summary(series)}


@app.get("/api/weather/forecast")
def weather_forecast(boundary_id: str, days: int = 7, user=Depends(need(feature="suitability"))):
    days = max(1, min(int(days), 16))
    _, lat, lon = _boundary_geometry(user, boundary_id)
    return {"boundary_id": boundary_id, "lat": lat, "lon": lon,
            **_weather_or_http_error(weather_mod.get_forecast, lat, lon, days)}


@app.get("/api/suitability/crops")
def suitability_crops(user=Depends(reader)):
    return [{"id": cid, "label": c["label"], "cycle_days": c["cycle_days"]}
            for cid, c in suitability_mod.load_crops().items()]


@app.post("/api/suitability/analyze")
def suitability_analyze(boundary_id: str = Form(...), crops: str = Form(""), planting_date: str = Form(""),
                        years: int = Form(10), user=Depends(need(feature="suitability"))):
    """Ranks crops for the farm: soil + climate scoring, the limiting factor for each,
    the best planting windows, and management advice. `crops` is a comma-separated list
    (blank = all); `planting_date` is YYYY-MM-DD or MM-DD (blank = search the best window)."""
    years = max(1, min(int(years), 30))
    geometry, lat, lon = _boundary_geometry(user, boundary_id)
    profile = _soil_or_http_error(geometry)
    series = _weather_or_http_error(weather_mod.get_daily_history, lat, lon, years)
    crop_ids = [c.strip().lower() for c in crops.split(",") if c.strip()] or None
    try:
        result = suitability_mod.analyze(profile, series, crop_ids, planting_date or None, lat=lat, lon=lon)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"boundary_id": boundary_id, **result}


@app.get("/api/health")
def health():
    with db.tx() as conn:
        conn.execute("SELECT 1")
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Many farms: the organization's portfolio, farms from Geohub, scouting tasks to Geohub
# ---------------------------------------------------------------------------

def _geohub_token(request: Request) -> str:
    token = getattr(request.state, "at", None) or request.cookies.get(identity.AT)
    if not token:
        raise Problem(401, "Sign in again to reach Geohub.", code="unauthenticated")
    return token


def _geohub(request: Request, path: str, params: dict | None = None):
    """Geohub's field data, read as the person (only what they could open in Geohub)."""
    try:
        return identity.client().data(_geohub_token(request), path, params)
    except GeoEstateIDError as e:
        if e.status in (401, 403, 404):
            raise Problem(e.status if e.status != 401 else 403, str(e), code=e.code or "geohub")
        raise Problem(502, f"Geohub could not be reached ({e}). Try again in a minute.", code="geohub_unreachable")


def _geohub_forms(request: Request, template: str) -> list[dict]:
    out = []
    for p in _geohub(request, "projects"):
        for f in p.get("forms", []):
            if f.get("template") == template:
                out.append({"form_id": str(f["id"]), "name": f["name"], "project_id": str(p["id"]), "project_name": p["name"],
                            "features": f.get("features", 0)})
    return out


def _all_features(request: Request, form_id: str) -> list[dict]:
    rows, after = [], 0
    while True:
        page = _geohub(request, f"forms/{form_id}/features", {"after": after, "limit": 1000})
        rows += page["features"]
        if not page.get("next"):
            return rows
        after = page["next"]


@app.get("/api/portfolio", summary="Every farm of the organization with its latest health, alerts and open scouting tasks")
def portfolio(index: str = "ndvi", user=Depends(reader)):
    """Ranked weakest first: farms whose latest image shows a drop (alert) first, then by the latest mean value. Farms
    with no usable image yet come last. Sample data counts only where DEMO_TOOLS is on."""
    if index not in INDEX_REGISTRY:
        raise HTTPException(400, f"Unknown index '{index}'")
    with db.tx() as conn:
        farms = conn.execute(
            "SELECT b.id::text AS id, b.name, b.area_ha, b.details, b.source, ST_Y(ST_Centroid(b.geom)) AS lat, "
            "  ST_X(ST_Centroid(b.geom)) AS lng, ST_AsGeoJSON(ST_SimplifyPreserveTopology(b.geom, 0.00002), 6)::jsonb AS geometry, "
            "  (SELECT count(*) FROM scouting_tasks t WHERE t.boundary_id = b.id AND t.status = 'open') AS open_tasks "
            "FROM boundaries b WHERE b.org_id = %s ORDER BY b.created_at", (user["org_id"],)).fetchall()
        obs = conn.execute(
            "SELECT o.boundary_id::text AS boundary_id, o.index_name, o.obs_date, o.source, o.scene_id, o.cloud_cover, "
            "  o.valid_fraction, o.mean, o.median, o.std, o.p10, o.p90, o.pixel_count FROM observations o "
            "JOIN boundaries b ON b.id = o.boundary_id WHERE b.org_id = %s AND o.index_name = %s"
            + ("" if settings.demo_tools else " AND o.source <> 'demo'") + " ORDER BY o.obs_date",
            (user["org_id"], index)).fetchall()
    by_farm: dict[str, list] = {}
    for o in obs:
        by_farm.setdefault(o["boundary_id"], []).append(o)
    out = []
    for f in farms:
        t = monitoring_mod.analyze_trend(by_farm.get(f["id"], []))
        usable = [s for s in t["series"] if s["usable"]]
        last = usable[-1] if usable else None
        d = f["details"] or {}
        out.append({
            "id": f["id"], "name": f["name"], "area_ha": f["area_ha"], "source": f["source"], "lat": f["lat"], "lng": f["lng"],
            "geometry": f["geometry"], "farmer": d.get("farmer_name"), "community": d.get("community"), "crops": d.get("crops") or [],
            "latest": last and {"obs_date": last["obs_date"], "mean": round(last["mean"], 3), "source": last["source"],
                                "change_vs_prev": None if last["change_vs_prev"] is None else round(last["change_vs_prev"], 3)},
            "images": len(usable), "status": t["status"], "alert": bool(last and last.get("alert")),
            "alerts": len(t["alerts"]), "open_tasks": f["open_tasks"],
        })
    out.sort(key=lambda r: (r["latest"] is None, not r["alert"], r["latest"]["mean"] if r["latest"] else 0))
    for i, r in enumerate(out, 1):
        r["rank"] = i
    return {"index": index, "farms": out, "farm_count": len(out), "hectares": round(sum(r["area_ha"] for r in out), 2),
            "with_alerts": sum(1 for r in out if r["alert"]), "without_images": sum(1 for r in out if not r["latest"])}


@app.get("/api/geohub/forms", summary="The person's Geohub forms made from a FarmScan template")
def geohub_forms(request: Request, template: str, user=Depends(reader)):
    if template not in ("farm_enumeration", "farm_scouting"):
        raise HTTPException(400, "template is farm_enumeration or farm_scouting")
    return _geohub_forms(request, template)


FARMER_KEYS = ("farmer_name", "farmer_phone", "farmer_gender", "farmer_group", "community", "lga", "state", "crops",
               "planting_date", "declared_ha", "tenure", "water", "farm_name", "remarks")


@app.post("/api/enumeration/import", summary="Bring in the farms a field team mapped with a Geohub \"Farm enumeration\" form")
def import_enumerated_farms(request: Request, form_id: str = Form(...), user=Depends(writer)):
    """Each submitted or approved record with an area becomes a farm, with its farmer's details (records already
    brought in are skipped, so it can be run again as more come in). Drafts and records a reviewer sent back are left
    out. The plan's hectares apply: farms that would go over are reported, not added."""
    form_id = _uuid(form_id, "Form")
    form = next((f for f in _geohub_forms(request, "farm_enumeration") if f["form_id"] == form_id), None)
    if not form:
        raise Problem(404, "That Geohub form is not a Farm enumeration form you can open.", code="not_found")
    feats = _all_features(request, form_id)
    added, skipped, problems = [], 0, []
    limit = (user["limits"] or {}).get("hectares")
    with db.tx() as conn:
        conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (user["org_id"],))
        used = conn.execute("SELECT coalesce(sum(area_ha), 0) AS ha FROM boundaries WHERE org_id = %s", (user["org_id"],)).fetchone()["ha"]
        have = {r["source_feature"] for r in conn.execute(
            "SELECT source_feature::text AS source_feature FROM boundaries WHERE org_id = %s AND source_feature IS NOT NULL",
            (user["org_id"],)).fetchall()}
        for ft in feats:
            fid = str(ft["id"])
            if ft.get("status") not in ("submitted", "approved"):
                continue
            if fid in have:
                skipped += 1
                continue
            p = ft.get("properties") or {}
            label = p.get("farm_name") or (f"{p['farmer_name']}'s farm" if p.get("farmer_name") else f"Farm {ft.get('seq')}")
            g = ft.get("geometry") or {}
            if g.get("type") not in ("Polygon", "MultiPolygon"):
                problems.append({"record": ft.get("seq"), "farm": label, "problem": "has no area (not a walked boundary)"})
                continue
            try:
                geom = _valid_shape(g)
            except HTTPException:
                problems.append({"record": ft.get("seq"), "farm": label, "problem": "its boundary crosses itself; fix it in Geohub"})
                continue
            area_ha = round(geo.area_m2(geom) / 10000, 2)
            if limit is not None and used + area_ha > limit:
                problems.append({"record": ft.get("seq"), "farm": label,
                                 "problem": f"{area_ha:,.2f} ha would go over the plan's {limit:,} ha"})
                continue
            details = {k: p[k] for k in FARMER_KEYS if p.get(k) not in (None, "", [])}
            details |= {"collected_by": ft.get("created_by_name"), "geohub_record": ft.get("seq"), "geohub_form": form["name"]}
            if community := p.get("community"):
                label = f"{label}, {community}" if community not in label else label
            bid = str(uuid.uuid4())
            conn.execute("INSERT INTO boundaries (id, org_id, name, geojson, area_ha, created_by, details, source, source_feature, "
                         "source_form) VALUES (%s, %s, %s, %s, %s, %s, %s, 'geohub', %s, %s)",
                         (bid, user["org_id"], label[:200], Jsonb(g), area_ha, user["name"], Jsonb(details), fid, form_id))
            used += area_ha
            added.append({"boundary_id": bid, "name": label, "area_ha": area_ha})
    return {"form": form["name"], "added": added, "already_here": skipped, "problems": problems,
            "hectares_used": round(used, 2), "hectares_limit": limit}


@app.get("/api/settings")
def get_settings(user=Depends(reader)):
    with db.tx() as conn:
        row = conn.execute("SELECT scouting_form::text AS scouting_form, scouting_project::text AS scouting_project, "
                           "scouting_form_name, updated_by, updated_at::text AS updated_at FROM org_settings WHERE org_id = %s",
                           (user["org_id"],)).fetchone()
    return row or {"scouting_form": None, "scouting_project": None, "scouting_form_name": None}


@app.put("/api/settings/scouting", summary="The Geohub form (made from the Farm scouting template) scouts answer in")
def set_scouting_form(request: Request, form_id: str = Form(...), user=Depends(need("manager", write=True, feature="monitoring"))):
    form_id = _uuid(form_id, "Form")
    form = next((f for f in _geohub_forms(request, "farm_scouting") if f["form_id"] == form_id), None)
    if not form:
        raise Problem(404, "That Geohub form is not a Farm scouting form you can open.", code="not_found")
    with db.tx() as conn:
        conn.execute("INSERT INTO org_settings (org_id, scouting_form, scouting_project, scouting_form_name, updated_by) "
                     "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (org_id) DO UPDATE SET scouting_form = EXCLUDED.scouting_form, "
                     "scouting_project = EXCLUDED.scouting_project, scouting_form_name = EXCLUDED.scouting_form_name, "
                     "updated_by = EXCLUDED.updated_by, updated_at = now()",
                     (user["org_id"], form_id, form["project_id"], f"{form['name']} ({form['project_name']})", user["name"]))
    return get_settings(user)


def _task_link(project: str, form: str, farm: str, ref: str, seen: str, lat: float, lng: float) -> str:
    fill = base64.urlsafe_b64encode(json.dumps({"farm": farm, "task_ref": ref, "seen": seen}, ensure_ascii=False)
                                    .encode()).decode().rstrip("=")
    return f"{settings.geohub_url}/collect/{project}?form={form}&fill={fill}&at={lat:.6f},{lng:.6f}"


TASK_SQL = ("SELECT id::text AS id, ref, rank, lat, lng, area_m2, index_name, obs_date, seen, status, link, findings, "
            "found_at::text AS found_at, created_by, created_at::text AS created_at FROM scouting_tasks ")


@app.post("/api/scouting/tasks", summary="Send the farm's problem areas to scouts: one Geohub link per area")
def send_to_scouts(boundary_id: str = Form(...), index: str = Form("ndvi"), sensitivity: float = Form(1.5),
                   min_area_m2: float = Form(400.0), user=Depends(need("agronomist", write=True, feature="monitoring"))):
    """Finds the problem areas in the last processed image (as Find problem areas does) and makes a scouting task for
    each, replacing this farm's tasks still open. Each task has a link that opens the organization's Geohub scouting
    form with the farm, the task and what the satellite saw filled in, and the place marked; send the links to scouts
    (WhatsApp, SMS). Their answers come back with "Check scouts' answers"."""
    farm = _farm(user, boundary_id)
    with db.tx() as conn:
        st = conn.execute("SELECT scouting_form::text AS f, scouting_project::text AS p FROM org_settings WHERE org_id = %s",
                          (user["org_id"],)).fetchone()
    if not st or not st["f"]:
        raise Problem(409, "Choose your organization's Geohub scouting form first (a form made from the Farm scouting "
                           "template). A manager sets it once.", code="no_scouting_form")
    cached = _cached(farm["id"], index)
    try:
        result = monitoring_mod.find_problem_areas(cached["array"], cached["transform"], cached["crs"], index,
                                                   sensitivity=sensitivity, min_area_m2=min_area_m2)
    except ValueError as e:
        raise HTTPException(400, str(e))
    obs_date = cached.get("obs_date") or ""
    tasks = []
    with db.tx() as conn:
        conn.execute("UPDATE scouting_tasks SET status = 'cancelled' WHERE boundary_id = %s AND status = 'open'", (farm["id"],))
        for a in result["areas"]:
            tid = uuid.uuid4()
            ref = f"FS-{tid.hex[:6].upper()}"
            seen = monitoring_mod.scouting_note_text(a, index, obs_date)
            lat, lng = a["centroid"]["lat"], a["centroid"]["lng"]
            link = _task_link(st["p"], st["f"], farm["name"], ref, seen, lat, lng)
            conn.execute("INSERT INTO scouting_tasks (id, boundary_id, ref, rank, lat, lng, area_m2, index_name, obs_date, seen, link, "
                         "created_by) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                         (str(tid), farm["id"], ref, a["rank"], lat, lng, a.get("area_m2"), index, obs_date, seen, link,
                          user["name"]))
        tasks = conn.execute(TASK_SQL + "WHERE boundary_id = %s AND status = 'open' ORDER BY rank", (farm["id"],)).fetchall()
    return {"farm": farm["name"], "obs_date": obs_date, "tasks": tasks, "summary": result["summary"],
            "message": _whatsapp_text(farm["name"], tasks)}


def _whatsapp_text(farm: str, tasks: list[dict]) -> str:
    lines = [f"Scouting for {farm}: {len(tasks)} place{'s' if len(tasks) != 1 else ''} to check. Open each link on your phone, "
             "go to the marked place and record what you find."]
    lines += [f"{t['rank']}. {t['ref']}: {t['link']}" for t in tasks]
    return "\n".join(lines)


@app.get("/api/scouting/tasks", summary="A farm's scouting tasks and what the scouts found")
def scouting_tasks(boundary_id: str, include_closed: bool = True, user=Depends(reader)):
    farm = _farm(user, boundary_id)
    with db.tx() as conn:
        rows = conn.execute(TASK_SQL + "WHERE boundary_id = %s AND status <> 'cancelled'" + ("" if include_closed else " AND status = 'open'")
                            + " ORDER BY status = 'open' DESC, rank", (farm["id"],)).fetchall()
    open_ = [r for r in rows if r["status"] == "open"]
    return {"farm": farm["name"], "tasks": rows, "open": len(open_), "message": _whatsapp_text(farm["name"], open_) if open_ else None}


@app.post("/api/scouting/refresh", summary="Read the scouts' answers back from Geohub")
def scouting_refresh(request: Request, user=Depends(need(feature="monitoring"))):
    """Matches the records in the organization's Geohub scouting form to the tasks by their task reference. A record
    a reviewer sent back, or a draft, is not taken."""
    with db.tx() as conn:
        st = conn.execute("SELECT scouting_form::text AS f FROM org_settings WHERE org_id = %s", (user["org_id"],)).fetchone()
    if not st or not st["f"]:
        raise Problem(409, "No Geohub scouting form is chosen yet.", code="no_scouting_form")
    feats = [f for f in _all_features(request, st["f"]) if f.get("status") in ("submitted", "approved")]
    by_ref = {}
    for f in feats:
        ref = (f.get("properties") or {}).get("task_ref")
        if ref:
            by_ref[str(ref).strip()] = f                       # the latest record for a task wins (oldest first)
    updated = 0
    with db.tx() as conn:
        tasks = conn.execute("SELECT t.id::text AS id, t.ref, t.findings FROM scouting_tasks t JOIN boundaries b ON b.id = t.boundary_id "
                             "WHERE b.org_id = %s AND t.status <> 'cancelled' AND t.ref = ANY(%s)",
                             (user["org_id"], list(by_ref))).fetchall()
        for t in tasks:
            f = by_ref[t["ref"]]
            p = f.get("properties") or {}
            photos = p.get("photos") or []
            findings = {"visited": p.get("visited"), "causes": p.get("causes") or [], "severity": p.get("severity"),
                        "affected_pct": p.get("affected_pct"), "notes": p.get("notes"), "photos": len(photos) if isinstance(photos, list) else 0,
                        "scout": f.get("created_by_name"), "recorded_at": f.get("updated_at") or f.get("created_at"),
                        "reviewed": f.get("status") == "approved",
                        "at": (f.get("geometry") or {}).get("coordinates")}
            status = "unreachable" if p.get("visited") == "Could not reach it" else "found"
            if findings != t["findings"]:
                conn.execute("UPDATE scouting_tasks SET findings = %s, status = %s, geohub_feature = %s, found_at = now() WHERE id = %s",
                             (Jsonb(findings), status, str(f["id"]), t["id"]))
                updated += 1
    return {"records_read": len(feats), "tasks_updated": updated}


# ---------------------------------------------------------------------------
# The season: NDVI from every usable Sentinel-2 image, radar from Sentinel-1, and green-up / peak / harvest
# ---------------------------------------------------------------------------

_JOBS = ThreadPoolExecutor(max_workers=2, thread_name_prefix="season")


def _submit(fn, *args):
    """Run a long job off the request (tests run it in place)."""
    _JOBS.submit(fn, *args)


def _job_update(job_id: str, **kw) -> None:
    sets = ", ".join(f"{k} = %s" for k in kw)
    vals = [Jsonb(v) if k == "result" else v for k, v in kw.items()]
    with db.tx() as conn:
        conn.execute(f"UPDATE season_jobs SET {sets}" + (", finished_at = now()" if kw.get("status") in ("done", "failed") else "")
                     + " WHERE id = %s", (*vals, job_id))


def _run_season_job(job_id: str) -> None:
    try:
        with db.tx() as conn:
            job = conn.execute("SELECT j.*, b.geojson FROM season_jobs j JOIN boundaries b ON b.id = j.boundary_id WHERE j.id = %s",
                               (job_id,)).fetchone()
        if not job:
            return
        bid, geometry = str(job["boundary_id"]), job["geojson"]
        start, end = job["start_date"].isoformat(), job["end_date"].isoformat()
        _job_update(job_id, status="running", progress=2, message="Looking for images…")
        import stac_client
        result = {"optical": {"images": 0, "usable": 0, "failed": 0}, "radar": {"images": 0, "failed": 0}}
        optical = stac_client.covering_scenes(geometry, start, end) if job["optical"] else []
        radar = season_mod.radar_scenes(geometry, start, end) if job["radar"] else []
        total = max(1, len(optical) + len(radar))
        done = 0
        for sc in optical:
            try:
                bands, meta = stac_client.fetch_and_clip_bands(sc["hrefs"], geometry, ["red", "nir"], reflectance_offset=sc["offset"])
                arr = compute_index("ndvi", bands)
                vf = meta.get("valid_fraction")
                if np.isfinite(arr).any():
                    monitoring_mod.save_observation(bid, "ndvi", sc["date"], "live", arr, scene_id=sc["scene_id"],
                                                    cloud_cover=sc.get("cloud_cover"), valid_fraction=vf)
                    result["optical"]["images"] += 1
                    result["optical"]["usable"] += int((vf or 0) >= monitoring_mod.MIN_VALID_FRACTION)
            except Exception as e:                      # one bad scene does not stop the season
                log.warning("season job %s: optical %s failed: %s", job_id, sc.get("scene_id"), e)
                result["optical"]["failed"] += 1
            done += 1
            _job_update(job_id, progress=int(5 + 90 * done / total), message=f"Optical images: {done} of {len(optical)}")
        for sc in radar:
            try:
                vv, vh = season_mod.radar_arrays(sc, geometry)
                if np.isfinite(vv).any() and np.isfinite(vh).any():
                    field = np.isfinite(vv) | np.isfinite(vh)
                    ok = np.isfinite(vv) & np.isfinite(vh)
                    vf = float(ok.sum() / max(1, field.sum()))
                    monitoring_mod.save_observation(bid, "rvi", sc["date"], "sentinel-1", season_mod.rvi(vv, vh),
                                                    scene_id=sc["scene_id"], valid_fraction=vf)
                    monitoring_mod.save_observation(bid, "vh_db", sc["date"], "sentinel-1", season_mod.to_db(vh),
                                                    scene_id=sc["scene_id"], valid_fraction=vf)
                    result["radar"]["images"] += 1
            except Exception as e:
                log.warning("season job %s: radar %s failed: %s", job_id, sc.get("scene_id"), e)
                result["radar"]["failed"] += 1
            done += 1
            _job_update(job_id, progress=int(5 + 90 * done / total), message=f"Radar images: {done - len(optical)} of {len(radar)}")
        _job_update(job_id, status="done", progress=100, result=result,
                    message=f"{result['optical']['images']} optical ({result['optical']['usable']} clear enough) and "
                            f"{result['radar']['images']} radar images added.")
    except Exception as e:
        log.exception("season job %s failed", job_id)
        _job_update(job_id, status="failed", message=f"Could not build the season: {e}")


JOB_SQL = ("SELECT id::text AS id, boundary_id::text AS boundary_id, start_date::text AS start_date, end_date::text AS end_date, "
           "optical, radar, status, progress, message, result, created_by, created_at::text AS created_at, "
           "finished_at::text AS finished_at FROM season_jobs ")


@app.post("/api/season/build", summary="Fetch every usable satellite image of a farm over a date range (runs as a job)")
def build_season(boundary_id: str = Form(...), start_date: str = Form(None), end_date: str = Form(None),
                 optical: bool = Form(True), radar: bool = Form(True),
                 user=Depends(need("agronomist", write=True, feature="monitoring"))):
    """Sentinel-2 NDVI for every image that covers the farm (each pixel cloud-masked; images under 60% clear are kept
    but not used for the reading) and Sentinel-1 radar (RVI and VH) for every pass, into the farm's timeline. Up to
    400 days; default the last 9 months. Poll GET /api/season/jobs/{id}."""
    farm = _farm(user, boundary_id)
    try:
        end = date_cls.fromisoformat(end_date) if end_date else date_cls.today()
        start = date_cls.fromisoformat(start_date) if start_date else end - timedelta(days=270)
    except ValueError:
        raise HTTPException(400, "Dates are YYYY-MM-DD")
    if not (0 < (end - start).days <= 400):
        raise HTTPException(400, "The dates must run forwards, over at most 400 days")
    if not (optical or radar):
        raise HTTPException(400, "Choose optical, radar or both")
    job_id = str(uuid.uuid4())
    with db.tx() as conn:
        conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (farm["id"],))
        busy = conn.execute("SELECT 1 FROM season_jobs WHERE boundary_id = %s AND status IN ('queued', 'running')",
                            (farm["id"],)).fetchone()
        if busy:
            raise Problem(409, "This farm's season is already being built. Wait for it to finish.", code="busy")
        conn.execute("INSERT INTO season_jobs (id, org_id, boundary_id, start_date, end_date, optical, radar, created_by) "
                     "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                     (job_id, user["org_id"], farm["id"], start, end, optical, radar, user["name"]))
    _submit(_run_season_job, job_id)
    return season_job(job_id, user)


@app.get("/api/season/jobs/{job_id}")
def season_job(job_id: str, user=Depends(reader)):
    with db.tx() as conn:
        row = conn.execute(JOB_SQL + "WHERE id = %s AND org_id = %s", (_uuid(job_id, "Job"), user["org_id"])).fetchone()
    if not row:
        raise Problem(404, "Job not found", code="not_found")
    return row


@app.get("/api/season", summary="The farm's season: NDVI and radar timelines, and green-up, peak and harvest")
def season(boundary_id: str, start_date: str = None, end_date: str = None, user=Depends(need(feature="monitoring"))):
    """Read from the images in the farm's timeline (build it first). NDVI images under 60% cloud-free are left out;
    radar uses the farm's median RVI per pass. Sample data counts only where DEMO_TOOLS is on."""
    farm = _farm(user, boundary_id)
    lo, hi = (start_date or "0000-01-01")[:10], (end_date or "9999-12-31")[:10]

    def series(index: str, sources: tuple[str, ...]):
        rows = [o for o in monitoring_mod.list_observations(farm["id"], index) if o["source"] in sources and lo <= o["obs_date"] <= hi]
        return rows

    optical_sources = ("live", "demo") if settings.demo_tools else ("live",)
    nd = series("ndvi", optical_sources)
    rv = series("rvi", ("sentinel-1", "demo"))
    vh = series("vh_db", ("sentinel-1", "demo"))
    usable = [o for o in nd if (o["valid_fraction"] or 0) >= monitoring_mod.MIN_VALID_FRACTION and o["mean"] is not None]
    reading = season_mod.season_reading([(o["obs_date"], o["mean"]) for o in usable],
                                        [(o["obs_date"], o["median"]) for o in rv if o["median"] is not None],
                                        (farm.get("details") or {}).get("planting_date"))
    return {
        "farm": farm["name"], **reading,
        "series": {
            "ndvi": [{"date": o["obs_date"], "value": None if o["mean"] is None else round(o["mean"], 3),
                      "clear": round(o["valid_fraction"] or 0, 2), "used": o in usable} for o in nd],
            "rvi": [{"date": o["obs_date"], "value": None if o["median"] is None else round(o["median"], 3)} for o in rv],
            "vh_db": [{"date": o["obs_date"], "value": None if o["median"] is None else round(o["median"], 2)} for o in vh],
        },
        "cloudy_images": len(nd) - len(usable),
    }


# ---------------------------------------------------------------------------
# Weekly farm alerts
# ---------------------------------------------------------------------------

def _remember_licence(user: dict) -> None:
    """Keep what the organization's licence allows, for the alerts sent with nobody signed in."""
    with db.tx() as conn:
        conn.execute("INSERT INTO org_settings (org_id, org_name, features, licence, seen_at) VALUES (%s, %s, %s, %s, now()) "
                     "ON CONFLICT (org_id) DO UPDATE SET org_name = EXCLUDED.org_name, features = EXCLUDED.features, "
                     "licence = EXCLUDED.licence, seen_at = now()",
                     (user["org_id"], user["org_name"], Jsonb(user["features"]), user["licence"]))


def org_alerts(org_id: str, today: date_cls | None = None) -> dict:
    """This week's alerts for each of the organization's farms. Each source that cannot be reached is reported, and
    the others still run."""
    today = today or date_cls.today()
    with db.tx() as conn:
        farms = conn.execute("SELECT id::text AS id, name, geojson, ST_Y(ST_Centroid(geom)) AS lat, ST_X(ST_Centroid(geom)) AS lng "
                             "FROM boundaries WHERE org_id = %s ORDER BY name", (org_id,)).fetchall()
    unavailable: set[str] = set()
    wx_cache: dict = {}
    out = []
    for f in farms:
        found = []
        cell = (round(f["lat"], 1), round(f["lng"], 1))                 # one weather call per ~10 km cell
        if cell not in wx_cache:
            try:
                wx = weather_mod.get_recent_and_ahead(f["lat"], f["lng"])
                try:
                    climate = weather_mod.climate_summary(weather_mod.get_daily_history(f["lat"], f["lng"], 10))
                except weather_mod.WeatherUnavailable:
                    climate = None
                wx_cache[cell] = (wx, climate)
            except weather_mod.WeatherUnavailable:
                wx_cache[cell] = None
        if wx_cache[cell]:
            wx, climate = wx_cache[cell]
            found += alerts_mod.weather_alerts(wx, alerts_mod.wet_months_of(climate), today)
        else:
            unavailable.add("weather")
        try:
            fires = alerts_mod.fetch_fires(f["geojson"])
            if fires is None:
                unavailable.add("fire (FIRMS_MAP_KEY not set)")
            else:
                found += alerts_mod.fire_alerts(fires, f["geojson"])
        except Exception:
            unavailable.add("fire")
        for index, sources in (("ndvi", ("live",)), ("rvi", ("sentinel-1",))):
            obs = [o for o in monitoring_mod.list_observations(f["id"], index) if o["source"] in sources]
            if obs:
                found += alerts_mod.crop_alerts(monitoring_mod.analyze_trend(obs), index, today)
        if found:
            out.append({"boundary_id": f["id"], "name": f["name"], "alerts": found})
    return {"farms_checked": len(farms), "farms": out, "alert_count": sum(len(x["alerts"]) for x in out),
            "unavailable": sorted(unavailable), "checked_on": today.isoformat()}


@app.get("/api/alerts", summary="This week's alerts on the organization's farms (as the weekly email would send them)")
def get_alerts(user=Depends(need(feature="alerts"))):
    _remember_licence(user)
    with db.tx() as conn:
        sub = conn.execute("SELECT 1 FROM alert_subscribers WHERE org_id = %s AND user_id = %s", (user["org_id"], user["id"])).fetchone()
        last = conn.execute("SELECT sent_at::text AS sent_at, recipients FROM alert_runs WHERE org_id = %s ORDER BY sent_at DESC LIMIT 1",
                            (user["org_id"],)).fetchone()
    return org_alerts(user["org_id"]) | {"subscribed": bool(sub), "last_email": last, "email_set_up": mailer.configured()}


@app.post("/api/alerts/subscribe", summary="Email me the weekly alerts")
def alerts_subscribe(user=Depends(need(feature="alerts"))):
    _remember_licence(user)
    with db.tx() as conn:
        conn.execute("INSERT INTO alert_subscribers (org_id, user_id, email, name) VALUES (%s, %s, %s, %s) ON CONFLICT (org_id, user_id) "
                     "DO UPDATE SET email = EXCLUDED.email, name = EXCLUDED.name", (user["org_id"], user["id"], user["email"], user["name"]))
    return {"subscribed": True, "email": user["email"]}


@app.delete("/api/alerts/subscribe", summary="Stop emailing me the weekly alerts")
def alerts_unsubscribe(user=Depends(reader)):
    with db.tx() as conn:
        conn.execute("DELETE FROM alert_subscribers WHERE org_id = %s AND user_id = %s", (user["org_id"], user["id"]))
    return {"subscribed": False}


ALERT_EVERY = timedelta(days=7)


def send_due_alerts(now: datetime | None = None) -> list[str]:
    """Send the weekly email to each organization that is due: it has subscribers, its licence (as last seen, within 60
    days) includes alerts and is not expired or suspended, and its last email was a week ago or more. Returns the
    organizations sent to. Safe to run from several processes (an advisory lock per organization)."""
    now = now or datetime.now(timezone.utc)
    sent = []
    with db.tx() as conn:
        orgs = conn.execute(
            "SELECT s.org_id::text AS org_id, s.org_name FROM org_settings s WHERE s.features ? 'alerts' "
            "AND s.licence IN ('demo', 'active', 'grace') AND s.seen_at > %s "
            "AND EXISTS (SELECT 1 FROM alert_subscribers a WHERE a.org_id = s.org_id) "
            "AND NOT EXISTS (SELECT 1 FROM alert_runs r WHERE r.org_id = s.org_id AND r.sent_at > %s)",
            (now - timedelta(days=60), now - ALERT_EVERY + timedelta(hours=1))).fetchall()
    for o in orgs:
        with db.tx() as conn:
            if not conn.execute("SELECT pg_try_advisory_xact_lock(hashtext('alerts:' || %s))", (o["org_id"],)).fetchone()["pg_try_advisory_xact_lock"]:
                continue
            if conn.execute("SELECT 1 FROM alert_runs WHERE org_id = %s AND sent_at > %s",
                            (o["org_id"], now - ALERT_EVERY + timedelta(hours=1))).fetchone():
                continue
            subs = conn.execute("SELECT email FROM alert_subscribers WHERE org_id = %s", (o["org_id"],)).fetchall()
            result = org_alerts(o["org_id"], now.date())
            if result["farms"]:
                subject, body = alerts_mod.email_text(o["org_name"] or "your organization", result["farms"], settings.app_url)
                for s_ in subs:
                    try:
                        mailer.send(s_["email"], subject, body)
                    except Exception as e:
                        log.warning("alerts: could not email %s: %s", s_["email"], e)
            conn.execute("INSERT INTO alert_runs (org_id, farms, alerts, recipients) VALUES (%s, %s, %s, %s)",
                         (o["org_id"], len(result["farms"]), Jsonb(result), len(subs) if result["farms"] else 0))
            sent.append(o["org_id"])
    return sent


def _alerts_loop(stop: threading.Event) -> None:
    while not stop.wait(3600):
        try:
            if mailer.configured():
                send_due_alerts()
        except Exception:
            log.exception("weekly alerts run failed")


# ---------------------------------------------------------------------------
# Farm verification reports
# ---------------------------------------------------------------------------

@app.post("/api/reports/verification", summary="Issue a numbered farm verification report (PDF), kept as issued")
def issue_verification(boundary_id: str = Form(...), user=Depends(need("agronomist", write=True, feature="verification"))):
    farm = _farm(user, boundary_id)
    geom = shape(farm["geojson"])
    c = geom.centroid
    d = farm.get("details") or {}
    declared = d.get("declared_ha")
    seasons, n_opt, n_rad = [], 0, 0
    nd = [o for o in monitoring_mod.list_observations(farm["id"], "ndvi") if o["source"] == "live"
          and (o["valid_fraction"] or 0) >= monitoring_mod.MIN_VALID_FRACTION and o["mean"] is not None]
    rv = [o for o in monitoring_mod.list_observations(farm["id"], "rvi") if o["source"] == "sentinel-1" and o["median"] is not None]
    years = sorted({o["obs_date"][:4] for o in nd + rv})[-3:]
    for y in years:
        ny = [(o["obs_date"], o["mean"]) for o in nd if o["obs_date"].startswith(y)]
        ry = [(o["obs_date"], o["median"]) for o in rv if o["obs_date"].startswith(y)]
        n_opt, n_rad = n_opt + len(ny), n_rad + len(ry)
        r = season_mod.season_reading(ny, ry)
        why = None if r["based_on"] else " ".join(x for x in (r["optical"].get("reason"), r["radar"].get("reason")) if x)
        seasons.append({"year": int(y), "based_on": r["based_on"], "why": why,
                        **{k: r.get(k) for k in ("planting", "greenup", "peak", "end")}})
    try:
        hist = weather_mod.get_daily_history(c.y, c.x, 10)
        rain = reports_mod.rainfall(hist, weather_mod.climate_summary(hist))
    except weather_mod.WeatherUnavailable:
        rain = None
    issued_at, issued_on = reports_mod.issued_now()
    code = reports_mod.new_code()
    content = {
        "code": code, "issued_at": issued_at, "issued_on": issued_on, "org_name": user["org_name"], "issued_by": user["name"],
        "verify_url": f"{settings.app_url}/verify.html?code={code}",
        "farm": {"name": farm["name"], "area_ha": farm["area_ha"], "lat": c.y, "lng": c.x,
                 "declared_ha": declared, "declared_vs_measured_pct": round(100 * (float(declared) - farm["area_ha"]) / farm["area_ha"])
                 if declared else None, "community": d.get("community"), "lga": d.get("lga"), "state": d.get("state"),
                 "farmer": d.get("farmer_name"), "crops": d.get("crops"), "mapped_by": d.get("collected_by") if farm.get("source") == "geohub" else None},
        "geometry": farm["geojson"], "seasons": seasons, "images": {"optical": n_opt, "radar": n_rad},
        "rainfall": rain, "title": reports_mod.title_status(c.y, c.x),
    }
    rid = str(uuid.uuid4())
    with db.tx() as conn:
        conn.execute("INSERT INTO farm_reports (id, code, org_id, org_name, boundary_id, content, created_by) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                     (rid, code, user["org_id"], user["org_name"], farm["id"], Jsonb(content), user["name"]))
    return {"id": rid, "code": code, "pdf": f"/api/reports/{rid}/pdf", "seasons": len(seasons),
            "warnings": [w for w, bad in (("The farm's satellite timeline is empty: build the season first.", not seasons),
                                          ("Rainfall could not be fetched.", rain is None)) if bad]}


@app.get("/api/reports", summary="The organization's verification reports (optionally for one farm)")
def list_reports(boundary_id: str | None = None, user=Depends(reader)):
    q = ("SELECT id::text AS id, code, boundary_id::text AS boundary_id, content->'farm'->>'name' AS farm, created_by, "
         "created_at::text AS created_at FROM farm_reports WHERE org_id = %s")
    args = [user["org_id"]]
    if boundary_id:
        q += " AND boundary_id = %s"
        args.append(_farm(user, boundary_id)["id"])
    with db.tx() as conn:
        return conn.execute(q + " ORDER BY created_at DESC LIMIT 200", args).fetchall()


@app.get("/api/reports/{report_id}/pdf")
def report_pdf(report_id: str, user=Depends(reader)):
    with db.tx() as conn:
        row = conn.execute("SELECT code, content FROM farm_reports WHERE id = %s AND org_id = %s",
                           (_uuid(report_id, "Report"), user["org_id"])).fetchone()
    if not row:
        raise Problem(404, "Report not found", code="not_found")
    return StreamingResponse(io.BytesIO(reports_mod.build_pdf(row["content"])), media_type="application/pdf",
                             headers={"Content-Disposition": f'attachment; filename="{row["code"]}.pdf"'})


@app.get("/api/verify/{code}", summary="Anyone holding a report checks it is genuine (no sign-in)")
def verify_report(code: str):
    with db.tx() as conn:
        row = conn.execute("SELECT code, org_name, created_at::text AS created_at, content FROM farm_reports WHERE code = %s",
                           (code.strip().upper()[:20],)).fetchone()
    if not row:
        return {"valid": False, "code": code}
    f = row["content"]["farm"]
    return {"valid": True, "code": row["code"], "issued_to": row["org_name"], "issued_at": row["created_at"],
            "farm": f["name"], "area_ha": f["area_ha"], "where": ", ".join(x for x in (f.get("community"), f.get("lga"), f.get("state")) if x),
            "seasons": [s["year"] for s in row["content"]["seasons"] if s.get("based_on")]}


app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")
