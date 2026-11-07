import base64
import io
import json
import logging
import tempfile
import uuid
from contextlib import asynccontextmanager
from datetime import date as date_cls
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
import plots as plots_mod
import storage
import zones as zones_mod
import exports as exports_mod
import plant_count as plant_count_mod
import drone_ingest
import annotations as annotations_mod
import plot_import as plot_import_mod
import monitoring as monitoring_mod
import soil as soil_mod
import weather as weather_mod
import suitability as suitability_mod
from config import settings
from geoestate_id import GeoEstateIDError
from identity import Problem, current_user, need, reader, writer

log = logging.getLogger("farmscan")
MAX_UPLOAD = 1024 * 1024 * 1024          # a drone orthomosaic: up to 1 GB

# In-memory cache of the last computed index result per (boundary_id, index), so plot stats / zones / exports can
# reuse the raster instead of recomputing. Farm ids are checked against the person's organization before use.
_RESULT_CACHE: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(_app):
    db.start()
    soil_mod.fetch_from_bucket()
    yield
    db.pool.close()


app = FastAPI(title="FarmScan API", description="Satellite and drone farm health mapping, on the GeoEstate platform",
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
    sql = ("SELECT id::text AS id, name, geojson, area_ha, created_by, created_at::text AS created_at FROM boundaries "
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
            "created_by": row["created_by"], "created_at": row["created_at"]}


@app.get("/api/boundaries")
def list_boundaries(user=Depends(reader)):
    with db.tx() as conn:
        return conn.execute("SELECT id::text AS id, name, area_ha, created_by, created_at::text AS created_at FROM boundaries "
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


# ---------------------------------------------------------------------------
# Plot layout (regular grid, manual editing, SHP/KML import)
# ---------------------------------------------------------------------------

@app.post("/api/plots/generate-grid")
def generate_grid_plots(boundary_id: str = Form(...), rows: int = Form(None), cols: int = Form(None),
                        target_plot_count: int = Form(None), user=Depends(writer)):
    """Manual/regular grid plot layout: splits the farm boundary into a rows x cols
    grid of evenly-sized trial plots. Use this when you want a uniform layout
    regardless of what's actually visible in imagery (e.g. planning plots before
    planting). Plots that already exist elsewhere (QGIS, a trial-management
    system) can be brought in with /api/plots/import, and any plot can be
    adjusted with the edit/add/delete endpoints. Provide either rows+cols
    directly, or target_plot_count to auto-estimate a grid."""
    farm = _farm(user, boundary_id)
    boundary_id, boundary_geojson = farm["id"], farm["geojson"]

    if rows is None or cols is None:
        if target_plot_count is None:
            raise HTTPException(400, "Provide either rows+cols or target_plot_count")
        rows, cols = plots_mod.estimate_grid_from_plot_count(boundary_geojson, target_plot_count)

    if not (1 <= rows <= 200 and 1 <= cols <= 200):
        raise HTTPException(400, "Rows and columns must each be between 1 and 200")
    plot_list = plots_mod.generate_grid_plots(boundary_geojson, rows, cols)
    _replace_plots(boundary_id, plot_list)

    return {"rows": rows, "cols": cols, "plot_count": len(plot_list), "plots": plot_list}


def _replace_plots(boundary_id: str, plot_list: list[dict]) -> None:
    """Replace a farm's plots with a new set (grid or import)."""
    with db.tx() as conn:
        conn.execute("DELETE FROM plots WHERE boundary_id = %s", (boundary_id,))
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO plots (id, boundary_id, plot_id, row_idx, col_idx, geojson, area_m2) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                [(str(uuid.uuid4()), boundary_id, p["plot_id"], p["row"], p["col"], Jsonb(p["geometry"]), p["area_m2"])
                 for p in plot_list])


def _plot_rows(boundary_id: str) -> list[dict]:
    with db.tx() as conn:
        return conn.execute("SELECT plot_id, row_idx, col_idx, geojson, area_m2 FROM plots WHERE boundary_id = %s "
                            "ORDER BY plot_id", (boundary_id,)).fetchall()


@app.get("/api/plots/{boundary_id}")
def get_plots(boundary_id: str, user=Depends(reader)):
    rows = _plot_rows(_farm(user, boundary_id)["id"])
    return {
        "plot_count": len(rows),
        "plots": [{"plot_id": r["plot_id"], "row": r["row_idx"], "col": r["col_idx"], "geometry": r["geojson"],
                   "area_m2": r["area_m2"]} for r in rows],
    }


@app.post("/api/plots/{boundary_id}/edit")
def edit_plot(boundary_id: str, plot_id: int = Form(...), geojson: str = Form(...), user=Depends(writer)):
    """
    Advanced plot editing: replaces a single plot's geometry (e.g. after the user
    drags/reshapes it on the map) without regenerating the whole set. row/col
    labels are kept as-is; area_m2 is recomputed from the new geometry.
    """
    new_geom, area_m2 = _plot_geometry(geojson)
    with db.tx() as conn:
        bid = _farm(user, boundary_id, conn)["id"]
        done = conn.execute("UPDATE plots SET geojson = %s, area_m2 = %s WHERE boundary_id = %s AND plot_id = %s",
                            (Jsonb(new_geom), area_m2, bid, plot_id)).rowcount
    if not done:
        raise HTTPException(404, "Plot not found for this farm")
    return {"plot_id": plot_id, "geometry": new_geom, "area_m2": area_m2}


def _plot_geometry(geojson: str) -> tuple[dict, float]:
    try:
        new_geom = json.loads(geojson)
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid JSON in geojson field")
    if not isinstance(new_geom, dict) or new_geom.get("type") not in ("Polygon", "MultiPolygon"):
        raise HTTPException(400, "A plot must be a Polygon or MultiPolygon")
    return new_geom, round(geo.area_m2(_valid_shape(new_geom)), 2)


@app.post("/api/plots/{boundary_id}/add")
def add_plot(boundary_id: str, geojson: str = Form(...), row: int = Form(None), col: int = Form(None),
             user=Depends(writer)):
    """Advanced plot editing: manually adds one new plot (e.g. one the grid missed). Assigns the next plot_id."""
    new_geom, area_m2 = _plot_geometry(geojson)
    with db.tx() as conn:
        bid = _farm(user, boundary_id, conn)["id"]
        conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (bid,))
        m = conn.execute("SELECT max(plot_id) AS m FROM plots WHERE boundary_id = %s", (bid,)).fetchone()["m"]
        next_plot_id = 0 if m is None else m + 1
        conn.execute("INSERT INTO plots (id, boundary_id, plot_id, row_idx, col_idx, geojson, area_m2) "
                     "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                     (str(uuid.uuid4()), bid, next_plot_id, row, col, Jsonb(new_geom), area_m2))
    return {"plot_id": next_plot_id, "row": row, "col": col, "geometry": new_geom, "area_m2": area_m2}


@app.delete("/api/plots/{boundary_id}/{plot_id}")
def delete_plot(boundary_id: str, plot_id: int, user=Depends(writer)):
    """Advanced plot editing: removes a single plot (e.g. an edge plot that isn't part of the trial)."""
    with db.tx() as conn:
        bid = _farm(user, boundary_id, conn)["id"]
        done = conn.execute("DELETE FROM plots WHERE boundary_id = %s AND plot_id = %s", (bid, plot_id)).rowcount
    if not done:
        raise HTTPException(404, "Plot not found for this farm")
    return {"deleted": True, "plot_id": plot_id}


@app.post("/api/plots/import")
async def import_plots(boundary_id: str = Form(...), file: UploadFile = File(...), user=Depends(writer)):
    """
    Import existing plots from a SHP (zipped shapefile: .shp/.shx/.dbf/.prj) or
    KML file, replacing any current plots for this boundary — for trials where
    plot boundaries were already digitized elsewhere (e.g. in QGIS or a trial
    management system) rather than drawn in FarmScan.
    """
    boundary_id = _farm(user, boundary_id)["id"]

    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()
    if suffix not in (".zip", ".kml", ".shp"):
        raise HTTPException(400, "Upload a zipped shapefile (.zip), a bare .shp, or a .kml file")

    content = await file.read(50 * 1024 * 1024 + 1)
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(413, "The plot file is over 50 MB")
    try:
        plot_list = plot_import_mod.parse_plots_file(content, suffix)
    except Exception as e:
        raise HTTPException(400, f"Could not read plot file: {e}")

    if not plot_list:
        raise HTTPException(400, "No polygon features found in the uploaded file")

    _replace_plots(boundary_id, plot_list)
    return {"plot_count": len(plot_list), "plots": plot_list, "source": f"imported_{suffix.lstrip('.')}"}


def _compute_plot_stats(boundary_id: str, index: str) -> dict:
    """Per-plot statistics for an already-processed index (Solvi's 'extensive
    statistics for each plot in a single click'). Requires plots to have been
    generated via /api/plots/generate-grid and the index processed already."""
    cached = _cached(boundary_id, index)
    plot_rows = _plot_rows(boundary_id)
    if not plot_rows:
        raise HTTPException(400, "This farm has no plots yet — generate a grid, draw or import plots first.")

    idx_array = cached["array"]
    transform = cached["transform"]
    crs = cached["crs"]

    # build an in-memory single-band raster so we can rasterio.mask clip per plot
    profile = {
        "driver": "GTiff", "dtype": "float32", "nodata": np.nan,
        "width": idx_array.shape[1], "height": idx_array.shape[0],
        "count": 1, "crs": crs, "transform": transform,
    }

    results = []
    no_data_count = 0
    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(**profile) as ds:
            ds.write(idx_array.astype(np.float32), 1)

        with memfile.open() as ds:
            for p in plot_rows:
                plot_geom = p["geojson"]
                reason = None
                try:
                    clipped, _ = rio_mask(ds, [plot_geom], crop=True, nodata=np.nan)
                    plot_arr = clipped[0]
                    valid_mask = ~np.isnan(plot_arr)
                    stats = zonal_stats(plot_arr, valid_mask)
                    if stats["mean"] is None:
                        reason = ("Plot falls outside the processed boundary shape (inside its "
                                   "bounding box, but outside the polygon) — redraw the boundary "
                                   "to fully cover this plot, or re-run the index.")
                except Exception as e:
                    stats = {"mean": None, "min": None, "max": None, "std": None, "pixel_count": 0}
                    reason = f"Could not clip this plot against the processed raster ({e})"

                if reason:
                    no_data_count += 1

                results.append({
                    "plot_id": p["plot_id"], "row": p["row_idx"], "col": p["col_idx"],
                    "area_m2": p["area_m2"], **stats, "no_data_reason": reason,
                })

    return {
        "index": index, "plot_count": len(results), "plots": results,
        "no_data_count": no_data_count,
    }




def _cached(boundary_id: str, index: str) -> dict:
    cached = _RESULT_CACHE.get(f"{boundary_id}:{index}")
    if not cached:
        raise HTTPException(400, f"No processed '{index}' result for this farm — run the index first.")
    return cached


@app.post("/api/plots/zonal-stats")
def plots_zonal_stats(boundary_id: str = Form(...), index: str = Form(...), user=Depends(reader)):
    """Per-plot statistics for an already-processed index (mean/min/max/std/pixel_count per
    plot). Requires plots (generate-grid / import) and the index processed first."""
    return _compute_plot_stats(_farm(user, boundary_id)["id"], index)


@app.post("/api/plots/performance")
def plots_performance(boundary_id: str = Form(...), index: str = Form(...), user=Depends(reader)):
    """Ranks plots against the field baseline (z-score vs median plot), flags
    underperforming / watch plots, and reports field uniformity and the headroom between
    the median plot and the top quartile. Uses the cached index raster."""
    stats = _compute_plot_stats(_farm(user, boundary_id)["id"], index)
    try:
        ranked = monitoring_mod.rank_plots(stats["plots"])
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"index": index, **ranked}


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


@app.get("/api/export/shp/plots")
def export_plots_shp(boundary_id: str, user=Depends(reader)):
    boundary_id = _farm(user, boundary_id)["id"]
    rows = _plot_rows(boundary_id)
    if not rows:
        raise HTTPException(400, "This farm has no plots yet.")

    features = [
        {"geometry": r["geojson"], "plot_id": r["plot_id"],
         "row": r["row_idx"], "col": r["col_idx"], "area_m2": r["area_m2"]}
        for r in rows
    ]
    zip_bytes = exports_mod.geometries_to_shapefile_zip_bytes(features, layer_name="plots")
    return StreamingResponse(
        io.BytesIO(zip_bytes), media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="plots_{boundary_id[:8]}.zip"'},
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


@app.get("/api/export/excel/plots")
def export_plots_excel(boundary_id: str, index: str, user=Depends(reader)):
    """Re-runs the same zonal-stats logic as /api/plots/zonal-stats and returns it as .xlsx."""
    boundary_id = _farm(user, boundary_id)["id"]
    resp = _compute_plot_stats(boundary_id, index)
    plot_stats = resp["plots"]
    xlsx_bytes = exports_mod.plot_stats_to_excel_bytes(plot_stats, sheet_title=f"{index.upper()} Plot Stats")
    return StreamingResponse(
        io.BytesIO(xlsx_bytes),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="plot_stats_{index}_{boundary_id[:8]}.xlsx"'},
    )


@app.get("/api/export/pdf")
def export_pdf(boundary_id: str, index: str, user=Depends(reader)):
    farm = _farm(user, boundary_id)
    boundary_id, farm_name = farm["id"], farm["name"]
    cached = _cached(boundary_id, index)
    plot_count = len(_plot_rows(boundary_id))

    pdf_bytes = exports_mod.build_pdf_report_bytes(
        farm_name, cached["label"], cached["stats"], plot_count, overlay_png_bytes=cached["png_bytes"]
    )
    return StreamingResponse(
        io.BytesIO(pdf_bytes), media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="report_{index}_{boundary_id[:8]}.pdf"'},
    )


# ---------------------------------------------------------------------------
# Drone orthomosaic upload + plant counting
# ---------------------------------------------------------------------------

@app.post("/api/drone/upload")
async def upload_drone_image(file: UploadFile = File(...), user=Depends(need("agronomist", write=True, feature="drone"))):
    """Upload a stitched drone orthomosaic GeoTIFF (Pix4D/Agisoft/DroneDeploy/OpenDroneMap output). Kept in the
    platform bucket for the organization. Returns metadata so the person can confirm the bands before running plant
    counts on it."""
    if not (file.filename or "").lower().endswith((".tif", ".tiff")):
        raise HTTPException(400, "Please upload a GeoTIFF (.tif/.tiff) orthomosaic")
    file_id = str(uuid.uuid4())
    tmp = Path(tempfile.mkdtemp(prefix="farm-drone-")) / f"{file_id}.tif"
    size = 0
    with open(tmp, "wb") as f:
        while chunk := await file.read(8 * 1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD:
                tmp.unlink(missing_ok=True)
                raise HTTPException(413, "The orthomosaic is over 1 GB. Export it at a coarser resolution (2–5 cm/pixel is enough).")
            f.write(chunk)
    try:
        meta = drone_ingest.inspect_orthomosaic(str(tmp))
        preview = drone_ingest.generate_preview(str(tmp))
    except Exception as e:
        tmp.unlink(missing_ok=True)
        raise HTTPException(400, f"Could not read file as GeoTIFF: {e}")
    k = storage.key("orgs", user["org_id"], "drone", f"{file_id}.tif")
    storage.put_file(tmp, k, "image/tiff")
    with db.tx() as conn:
        conn.execute("INSERT INTO drone_files (id, org_id, filename, storage_key, size_bytes, uploaded_by) "
                     "VALUES (%s, %s, %s, %s, %s, %s)", (file_id, user["org_id"], file.filename[:200], k, size, user["name"]))
    return {"drone_file_id": file_id, **meta, **preview}


def _drone_file(user: dict, drone_file_id: str) -> str:
    """A local copy of one of the organization's orthomosaics."""
    with db.tx() as conn:
        row = conn.execute("SELECT storage_key FROM drone_files WHERE id = %s AND org_id = %s",
                           (_uuid(drone_file_id, "Drone file"), user["org_id"])).fetchone()
    path = storage.local_path(row["storage_key"]) if row else None
    if not path:
        raise HTTPException(404, "Uploaded drone file not found — upload the orthomosaic again.")
    return str(path)


@app.post("/api/drone/plant-count")
def drone_plant_count(drone_file_id: str = Form(...), boundary_id: str = Form(None), min_plant_area_px: int = Form(15),
                      user=Depends(need(feature="plant_count"))):
    """
    Runs blob/watershed-based plant detection on an uploaded drone orthomosaic.
    Requires centimeter-scale imagery — will not work meaningfully on satellite data.
    See plant_count.py docstring for accuracy caveats (baseline method, not a
    trained model — best on widely-spaced row/nursery crops).
    """
    file_path = _drone_file(user, drone_file_id)
    boundary_geojson = _farm(user, boundary_id)["geojson"] if boundary_id else None

    rgb, meta = drone_ingest.load_rgb_for_plant_count(file_path, boundary_geojson)
    result = plant_count_mod.count_plants(rgb, min_plant_area_px=min_plant_area_px)
    geo_points = plant_count_mod.pixel_centroids_to_geojson_points(
        result["centroids_px"], result["areas_px"], meta["transform"], meta["crs"]
    )

    return {
        "plant_count": result["count"],
        "mean_canopy_area_px": result["mean_canopy_area_px"],
        "plants": geo_points,
        "note": "Baseline blob/watershed detector — best accuracy on widely-spaced row or "
                "nursery crops with cm-resolution imagery. For dense closed-canopy fields, "
                "accuracy will be lower without a crop-specific trained model.",
    }


@app.post("/api/drone/plant-count-per-plot")
def drone_plant_count_per_plot(drone_file_id: str = Form(...), boundary_id: str = Form(...),
                               min_plant_area_px: int = Form(15), user=Depends(need(feature="plant_count"))):
    """
    Plant counts broken down per plot, rather than one field-wide total. Requires plots to already exist for this
    farm (from a grid, drawing or import).
    """
    file_path = _drone_file(user, drone_file_id)
    farm = _farm(user, boundary_id)
    plot_rows = _plot_rows(farm["id"])
    if not plot_rows:
        raise HTTPException(400, "This farm has no plots yet — generate a grid, draw or import plots first.")
    plots = [{"plot_id": r["plot_id"], "row": r["row_idx"], "col": r["col_idx"], "geometry": r["geojson"]}
             for r in plot_rows]

    rgb, meta = drone_ingest.load_rgb_for_plant_count(file_path, farm["geojson"])
    result = plant_count_mod.count_plants(rgb, min_plant_area_px=min_plant_area_px)
    geo_points = plant_count_mod.pixel_centroids_to_geojson_points(
        result["centroids_px"], result["areas_px"], meta["transform"], meta["crs"]
    )
    per_plot = plant_count_mod.aggregate_plants_per_plot(geo_points, plots)

    return {
        "boundary_id": farm["id"],
        "total_plant_count": result["count"],
        "plot_count": len(per_plot),
        "plots": per_plot,
    }


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


app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")
