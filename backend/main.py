import base64
import io
import json
import sqlite3
import uuid
from pathlib import Path

import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from shapely.geometry import shape, mapping

from indices import INDEX_REGISTRY, compute_index, zonal_stats
import demo_data
import plots as plots_mod
import zones as zones_mod
import exports as exports_mod
import plant_count as plant_count_mod
import drone_ingest
import annotations as annotations_mod
import plot_import as plot_import_mod
import sam_plots as sam_plots_mod

DB_PATH = Path(__file__).parent / "farmscan.db"
UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# In-memory cache of the last computed index result per (boundary_id, index),
# so plot stats / zones / exports can reuse the raster instead of recomputing.
# Keyed as f"{boundary_id}:{index}" -> {"array": np.ndarray, "transform":, "crs":,
#                                        "png_bytes": bytes, "stats": dict}
_RESULT_CACHE: dict[str, dict] = {}

app = FastAPI(title="FarmScan API", description="Open-satellite farm health mapping")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS boundaries (
            id TEXT PRIMARY KEY,
            name TEXT,
            geojson TEXT NOT NULL,
            area_ha REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS plots (
            id TEXT PRIMARY KEY,
            boundary_id TEXT NOT NULL,
            plot_id INTEGER,
            row_idx INTEGER,
            col_idx INTEGER,
            geojson TEXT NOT NULL,
            area_m2 REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


init_db()
annotations_mod.init_sharing_tables()


# ---------------------------------------------------------------------------
# Boundary import
# ---------------------------------------------------------------------------

@app.post("/api/boundary")
async def create_boundary(name: str = Form(...), geojson: str = Form(...)):
    """
    Import a farm/field boundary. `geojson` is a stringified GeoJSON Feature,
    FeatureCollection, or bare Polygon/MultiPolygon geometry (EPSG:4326).
    """
    try:
        parsed = json.loads(geojson)
    except json.JSONDecodeError:
        raise HTTPException(400, "Invalid JSON in geojson field")

    # normalize to a bare geometry dict
    if parsed.get("type") == "FeatureCollection":
        geometry = parsed["features"][0]["geometry"]
    elif parsed.get("type") == "Feature":
        geometry = parsed["geometry"]
    else:
        geometry = parsed

    try:
        geom = shape(geometry)
        if not geom.is_valid or geom.is_empty:
            raise ValueError("Geometry is invalid or empty")
    except Exception as e:
        raise HTTPException(400, f"Could not parse geometry: {e}")

    # rough area in hectares using an equal-area-ish approximation (good enough for display;
    # for production use a proper projected CRS transform based on the boundary's location)
    area_deg2 = geom.area
    lat = geom.centroid.y
    m_per_deg_lat = 111320
    m_per_deg_lon = 111320 * np.cos(np.radians(lat))
    area_m2 = area_deg2 * m_per_deg_lat * m_per_deg_lon
    area_ha = round(area_m2 / 10000, 2)

    boundary_id = str(uuid.uuid4())
    conn = get_db()
    conn.execute(
        "INSERT INTO boundaries (id, name, geojson, area_ha) VALUES (?, ?, ?, ?)",
        (boundary_id, name, json.dumps(geometry), area_ha),
    )
    conn.commit()
    conn.close()

    return {
        "boundary_id": boundary_id,
        "name": name,
        "area_ha": area_ha,
        "centroid": [geom.centroid.y, geom.centroid.x],
        "bounds": list(geom.bounds),  # [minx, miny, maxx, maxy]
    }


@app.get("/api/boundary/{boundary_id}")
async def get_boundary(boundary_id: str):
    conn = get_db()
    row = conn.execute("SELECT * FROM boundaries WHERE id = ?", (boundary_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "Boundary not found")
    return {
        "boundary_id": row["id"],
        "name": row["name"],
        "geojson": json.loads(row["geojson"]),
        "area_ha": row["area_ha"],
        "created_at": row["created_at"],
    }


@app.get("/api/boundaries")
async def list_boundaries():
    conn = get_db()
    rows = conn.execute("SELECT id, name, area_ha, created_at FROM boundaries ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(r) for r in rows]


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
async def search_imagery(boundary_id: str, start_date: str, end_date: str, max_cloud_cover: float = 30.0):
    """Search Sentinel-2 scenes covering the boundary. Requires outbound network access
    to planetarycomputer.microsoft.com — falls back with a clear error if unreachable."""
    conn = get_db()
    row = conn.execute("SELECT geojson FROM boundaries WHERE id = ?", (boundary_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "Boundary not found")

    try:
        import stac_client
        geometry = json.loads(row["geojson"])
        scenes = stac_client.search_scenes(geometry, start_date, end_date, max_cloud_cover)
        # strip the href cache before sending to client; keep it server-side keyed by scene_id
        _SCENE_CACHE.update({s["scene_id"]: s["_href_cache"] for s in scenes})
        for s in scenes:
            s.pop("_href_cache", None)
        return {"scenes": scenes, "source": "live"}
    except Exception as e:
        raise HTTPException(
            502,
            f"Could not reach satellite catalog ({e}). "
            "Use /api/imagery/process-demo to test the pipeline with synthetic data instead."
        )


_SCENE_CACHE: dict[str, dict] = {}  # scene_id -> band href map, populated by search_imagery


# ---------------------------------------------------------------------------
# Index processing (live)
# ---------------------------------------------------------------------------

@app.post("/api/imagery/process")
async def process_imagery(boundary_id: str = Form(...), scene_id: str = Form(...), index: str = Form(...)):
    if index not in INDEX_REGISTRY:
        raise HTTPException(400, f"Unknown index '{index}'. Choose from {list(INDEX_REGISTRY)}")

    href_map = _SCENE_CACHE.get(scene_id)
    if not href_map:
        raise HTTPException(400, "Unknown scene_id — call /api/imagery/search first")

    conn = get_db()
    row = conn.execute("SELECT geojson FROM boundaries WHERE id = ?", (boundary_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "Boundary not found")
    boundary_geojson = json.loads(row["geojson"])

    import stac_client
    bands_needed = INDEX_REGISTRY[index]["bands"]
    band_arrays, out_meta = stac_client.fetch_and_clip_bands(href_map, boundary_geojson, bands_needed)

    return _finish_processing(index, band_arrays, out_meta, boundary_geojson, boundary_id=boundary_id)


# ---------------------------------------------------------------------------
# Index processing (demo / offline mode — synthetic data, no network needed)
# ---------------------------------------------------------------------------

@app.post("/api/imagery/process-demo")
async def process_imagery_demo(boundary_id: str = Form(...), index: str = Form(...)):
    if index not in INDEX_REGISTRY:
        raise HTTPException(400, f"Unknown index '{index}'. Choose from {list(INDEX_REGISTRY)}")

    conn = get_db()
    row = conn.execute("SELECT geojson FROM boundaries WHERE id = ?", (boundary_id,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "Boundary not found")
    boundary_geojson = json.loads(row["geojson"])

    band_arrays, out_meta, field_mask = demo_data.synthetic_bands_for_boundary(boundary_geojson)
    return _finish_processing(index, band_arrays, out_meta, boundary_geojson, field_mask, demo=True,
                               boundary_id=boundary_id)


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
                        boundary_id: str = None):
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
        }

    return JSONResponse({
        "index": index,
        "label": spec["label"],
        "demo_mode": demo,
        "overlay_png_base64": png_b64,
        "overlay_bounds": [[miny, minx], [maxy, maxx]],  # Leaflet [[south, west], [north, east]]
        "value_range": spec["range"],
        "colormap": spec["colormap"],
        "stats": stats,
    })


# ---------------------------------------------------------------------------
# Plot extraction (automatic grid, or field-boundary auto-detect)
# ---------------------------------------------------------------------------

@app.post("/api/plots/generate-grid")
async def generate_grid_plots(boundary_id: str = Form(...), rows: int = Form(None),
                               cols: int = Form(None), target_plot_count: int = Form(None)):
    """Manual/regular grid plot layout: splits the farm boundary into a rows x cols
    grid of evenly-sized trial plots. Use this when you want a uniform layout
    regardless of what's actually visible in imagery (e.g. planning plots before
    planting). For plots detected automatically from an uploaded drone image's
    actual planted areas, use /api/plots/extract-from-drone instead. Provide
    either rows+cols directly, or target_plot_count to auto-estimate a grid."""
    conn = get_db()
    row = conn.execute("SELECT geojson FROM boundaries WHERE id = ?", (boundary_id,)).fetchone()
    if not row:
        conn.close()
        raise HTTPException(404, "Boundary not found")
    boundary_geojson = json.loads(row["geojson"])

    if rows is None or cols is None:
        if target_plot_count is None:
            raise HTTPException(400, "Provide either rows+cols or target_plot_count")
        rows, cols = plots_mod.estimate_grid_from_plot_count(boundary_geojson, target_plot_count)

    plot_list = plots_mod.generate_grid_plots(boundary_geojson, rows, cols)

    # clear any previously saved plots for this boundary, then save the new set
    conn.execute("DELETE FROM plots WHERE boundary_id = ?", (boundary_id,))
    for p in plot_list:
        conn.execute(
            "INSERT INTO plots (id, boundary_id, plot_id, row_idx, col_idx, geojson, area_m2) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), boundary_id, p["plot_id"], p["row"], p["col"],
             json.dumps(p["geometry"]), p["area_m2"]),
        )
    conn.commit()
    conn.close()

    return {"rows": rows, "cols": cols, "plot_count": len(plot_list), "plots": plot_list}


@app.post("/api/plots/extract-from-drone")
async def extract_plots_from_drone(drone_file_id: str = Form(...), boundary_id: str = Form(None),
                                    min_area_px: int = Form(300), method: str = Form("vegetation")):
    """
    Automatic plot extraction: detects individual trial plots directly from an
    uploaded drone orthomosaic (via /api/drone/upload) — no manual grid is
    drawn. Each distinct area in the image becomes its own plot, with a
    best-effort row/col label assigned from plot positions. If boundary_id is
    given, results are clipped to that boundary and saved so per-plot stats /
    exports work as usual; otherwise the detected plots are returned without
    being persisted.

    method:
      - "vegetation" (default): classical excess-green vegetation-mask +
        contour detection (plots.py). Fast, no extra dependencies. Assumes
        plots are distinguished from their surroundings by canopy greenness
        — works well for young/sparse trial plots on bare soil, poorly when
        plots are instead delineated by roads/tracks cutting through
        continuous vegetation.
      - "sam": zero-shot segmentation using a pretrained model (MobileSAM,
        see sam_plots.py) instead of a hand-tuned color rule. Better at
        separating visually-distinct regions in general (including
        road-delineated plots), but needs torch+timm installed and is much
        slower on CPU. Falls back with a clear error if unavailable.
    """
    file_path = UPLOAD_DIR / f"{drone_file_id}.tif"
    if not file_path.exists():
        raise HTTPException(404, "Uploaded drone file not found — upload it via /api/drone/upload first.")

    boundary_geojson = None
    if boundary_id:
        conn = get_db()
        row = conn.execute("SELECT geojson FROM boundaries WHERE id = ?", (boundary_id,)).fetchone()
        conn.close()
        if row:
            boundary_geojson = json.loads(row["geojson"])

    rgb, meta = drone_ingest.load_rgb_for_plant_count(str(file_path), boundary_geojson)

    if method == "sam":
        ok, reason = sam_plots_mod.sam_available()
        if not ok:
            raise HTTPException(
                400, f"SAM extraction method unavailable ({reason}). Use method=vegetation instead."
            )
        try:
            plot_list = sam_plots_mod.extract_plots_via_sam(rgb, meta["transform"], meta["crs"])
        except Exception as e:
            raise HTTPException(500, f"SAM extraction failed: {e}")
    else:
        veg_mask = plant_count_mod.excess_green_mask(rgb)
        plot_list = plots_mod.extract_plots_from_drone_mask(
            veg_mask, meta["transform"], meta["crs"], min_area_px=min_area_px
        )

    if boundary_id and plot_list:
        conn = get_db()
        conn.execute("DELETE FROM plots WHERE boundary_id = ?", (boundary_id,))
        for p in plot_list:
            conn.execute(
                "INSERT INTO plots (id, boundary_id, plot_id, row_idx, col_idx, geojson, area_m2) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (str(uuid.uuid4()), boundary_id, p["plot_id"], p["row"], p["col"],
                 json.dumps(p["geometry"]), p["area_m2"]),
            )
        conn.commit()
        conn.close()

    return {
        "plot_count": len(plot_list),
        "plots": plot_list,
        "source": "drone_image_auto_detect",
        "saved": bool(boundary_id and plot_list),
    }


@app.get("/api/plots/extraction-methods")
async def extraction_methods():
    """Reports which plot-extraction methods are currently usable, so the
    frontend can grey out 'sam' if torch/timm/weights aren't installed rather
    than letting the user hit a 400 after uploading a file."""
    sam_ok, sam_reason = sam_plots_mod.sam_available()
    return {
        "vegetation": {"available": True, "reason": None},
        "sam": {"available": sam_ok, "reason": None if sam_ok else sam_reason},
    }


@app.get("/api/plots/{boundary_id}")
async def get_plots(boundary_id: str):
    conn = get_db()
    rows = conn.execute(
        "SELECT plot_id, row_idx, col_idx, geojson, area_m2 FROM plots WHERE boundary_id = ? "
        "ORDER BY plot_id", (boundary_id,)
    ).fetchall()
    conn.close()
    return {
        "plot_count": len(rows),
        "plots": [
            {"plot_id": r["plot_id"], "row": r["row_idx"], "col": r["col_idx"],
             "geometry": json.loads(r["geojson"]), "area_m2": r["area_m2"]}
            for r in rows
        ],
    }


@app.post("/api/plots/{boundary_id}/edit")
async def edit_plot(boundary_id: str, plot_id: int = Form(...), geojson: str = Form(...)):
    """
    Advanced plot editing: replaces a single plot's geometry (e.g. after the user
    drags/reshapes it on the map) without regenerating the whole set. row/col
    labels are kept as-is; area_m2 is recomputed from the new geometry.
    """
    try:
        new_geom = json.loads(geojson)
        geom_obj = shape(new_geom)
        if not geom_obj.is_valid or geom_obj.is_empty:
            raise ValueError("Geometry is invalid or empty")
    except Exception as e:
        raise HTTPException(400, f"Could not parse geometry: {e}")

    lat = geom_obj.centroid.y
    m_per_deg_lat = 111320
    m_per_deg_lon = 111320 * np.cos(np.radians(lat))
    area_m2 = round(geom_obj.area * m_per_deg_lat * m_per_deg_lon, 2)

    conn = get_db()
    existing = conn.execute(
        "SELECT id FROM plots WHERE boundary_id = ? AND plot_id = ?", (boundary_id, plot_id)
    ).fetchone()
    if not existing:
        conn.close()
        raise HTTPException(404, "Plot not found for this boundary")

    conn.execute(
        "UPDATE plots SET geojson = ?, area_m2 = ? WHERE boundary_id = ? AND plot_id = ?",
        (json.dumps(new_geom), area_m2, boundary_id, plot_id),
    )
    conn.commit()
    conn.close()
    return {"plot_id": plot_id, "geometry": new_geom, "area_m2": area_m2}


@app.post("/api/plots/{boundary_id}/add")
async def add_plot(boundary_id: str, geojson: str = Form(...), row: int = Form(None), col: int = Form(None)):
    """Advanced plot editing: manually adds one new plot (e.g. the user draws a plot
    the automatic/grid tools missed). Assigns the next available plot_id."""
    try:
        new_geom = json.loads(geojson)
        geom_obj = shape(new_geom)
        if not geom_obj.is_valid or geom_obj.is_empty:
            raise ValueError("Geometry is invalid or empty")
    except Exception as e:
        raise HTTPException(400, f"Could not parse geometry: {e}")

    lat = geom_obj.centroid.y
    m_per_deg_lat = 111320
    m_per_deg_lon = 111320 * np.cos(np.radians(lat))
    area_m2 = round(geom_obj.area * m_per_deg_lat * m_per_deg_lon, 2)

    conn = get_db()
    max_row = conn.execute(
        "SELECT MAX(plot_id) as m FROM plots WHERE boundary_id = ?", (boundary_id,)
    ).fetchone()
    next_plot_id = (max_row["m"] + 1) if max_row["m"] is not None else 0

    conn.execute(
        "INSERT INTO plots (id, boundary_id, plot_id, row_idx, col_idx, geojson, area_m2) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (str(uuid.uuid4()), boundary_id, next_plot_id, row, col, json.dumps(new_geom), area_m2),
    )
    conn.commit()
    conn.close()
    return {"plot_id": next_plot_id, "row": row, "col": col, "geometry": new_geom, "area_m2": area_m2}


@app.delete("/api/plots/{boundary_id}/{plot_id}")
async def delete_plot(boundary_id: str, plot_id: int):
    """Advanced plot editing: removes a single plot (e.g. a false-positive from
    automatic detection, or an edge plot that isn't part of the trial)."""
    conn = get_db()
    existing = conn.execute(
        "SELECT id FROM plots WHERE boundary_id = ? AND plot_id = ?", (boundary_id, plot_id)
    ).fetchone()
    if not existing:
        conn.close()
        raise HTTPException(404, "Plot not found for this boundary")
    conn.execute("DELETE FROM plots WHERE boundary_id = ? AND plot_id = ?", (boundary_id, plot_id))
    conn.commit()
    conn.close()
    return {"deleted": True, "plot_id": plot_id}


@app.post("/api/plots/import")
async def import_plots(boundary_id: str = Form(...), file: UploadFile = File(...)):
    """
    Import existing plots from a SHP (zipped shapefile: .shp/.shx/.dbf/.prj) or
    KML file, replacing any current plots for this boundary — for trials where
    plot boundaries were already digitized elsewhere (e.g. in QGIS or a trial
    management system) rather than generated/detected in FarmScan.
    """
    conn = get_db()
    boundary_row = conn.execute("SELECT id FROM boundaries WHERE id = ?", (boundary_id,)).fetchone()
    conn.close()
    if not boundary_row:
        raise HTTPException(404, "Boundary not found")

    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()
    if suffix not in (".zip", ".kml", ".shp"):
        raise HTTPException(400, "Upload a zipped shapefile (.zip), a bare .shp, or a .kml file")

    content = await file.read()
    try:
        plot_list = plot_import_mod.parse_plots_file(content, suffix)
    except Exception as e:
        raise HTTPException(400, f"Could not read plot file: {e}")

    if not plot_list:
        raise HTTPException(400, "No polygon features found in the uploaded file")

    conn = get_db()
    conn.execute("DELETE FROM plots WHERE boundary_id = ?", (boundary_id,))
    for p in plot_list:
        conn.execute(
            "INSERT INTO plots (id, boundary_id, plot_id, row_idx, col_idx, geojson, area_m2) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), boundary_id, p["plot_id"], p["row"], p["col"],
             json.dumps(p["geometry"]), p["area_m2"]),
        )
    conn.commit()
    conn.close()

    return {"plot_count": len(plot_list), "plots": plot_list, "source": f"imported_{suffix.lstrip('.')}"}


@app.post("/api/plots/detect-boundary")
async def detect_boundary(boundary_id: str = Form(...), index: str = Form("ndvi"),
                           threshold: float = Form(0.3)):
    """Automatic field-boundary detection from an already-processed vegetation index
    raster (run /api/imagery/process-demo or /process for this index+boundary first).
    Returns candidate polygons the user can accept/edit — useful when you have
    imagery but haven't digitized a boundary yet."""
    cache_key = f"{boundary_id}:{index}"
    cached = _RESULT_CACHE.get(cache_key)
    if not cached:
        raise HTTPException(
            400, f"No processed '{index}' result cached for this boundary yet — "
                 f"run /api/imagery/process-demo (or /process) with index='{index}' first."
        )

    veg_mask = plots_mod.veg_mask_from_ndvi(cached["array"], threshold=threshold)
    candidates = plots_mod.detect_field_boundary(veg_mask, cached["transform"])

    return {"candidate_count": len(candidates), "candidates": candidates}


@app.post("/api/plots/zonal-stats")
async def plots_zonal_stats(boundary_id: str = Form(...), index: str = Form(...)):
    """Per-plot statistics for an already-processed index (Solvi's 'extensive
    statistics for each plot in a single click'). Requires plots to have been
    generated via /api/plots/generate-grid and the index processed already."""
    cache_key = f"{boundary_id}:{index}"
    cached = _RESULT_CACHE.get(cache_key)
    if not cached:
        raise HTTPException(400, f"No processed '{index}' result cached — process it first.")

    conn = get_db()
    plot_rows = conn.execute(
        "SELECT plot_id, row_idx, col_idx, geojson, area_m2 FROM plots WHERE boundary_id = ? ORDER BY plot_id",
        (boundary_id,),
    ).fetchall()
    conn.close()
    if not plot_rows:
        raise HTTPException(400, "No plots found — call /api/plots/generate-grid first.")

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
                plot_geom = json.loads(p["geojson"])
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


# ---------------------------------------------------------------------------
# Management zones & prescription export
# ---------------------------------------------------------------------------

@app.post("/api/zones/classify")
async def classify_zones_endpoint(boundary_id: str = Form(...), index: str = Form(...),
                                   n_zones: int = Form(3)):
    """Clusters the index raster into n management zones (e.g. 'low/medium/high
    vigor') and polygonizes them — the input for a variable-rate prescription file."""
    cache_key = f"{boundary_id}:{index}"
    cached = _RESULT_CACHE.get(cache_key)
    if not cached:
        raise HTTPException(400, f"No processed '{index}' result cached — process it first.")

    try:
        zone_array = zones_mod.classify_zones(cached["array"], n_zones=n_zones)
        zone_polygons = zones_mod.zones_to_polygons(zone_array, cached["transform"], cached["crs"], n_zones)
    except ValueError as e:
        raise HTTPException(400, str(e))

    # cache for export
    _RESULT_CACHE[f"{boundary_id}:{index}:zones:{n_zones}"] = {"zones": zone_polygons}

    return {"index": index, "n_zones": n_zones, "zones": zone_polygons}


# ---------------------------------------------------------------------------
# Exports: GeoTIFF, Shapefile, Excel, PDF
# ---------------------------------------------------------------------------

@app.get("/api/export/geotiff")
async def export_geotiff(boundary_id: str, index: str):
    cached = _RESULT_CACHE.get(f"{boundary_id}:{index}")
    if not cached:
        raise HTTPException(400, f"No processed '{index}' result cached — process it first.")
    tiff_bytes = exports_mod.index_array_to_geotiff_bytes(cached["array"], cached["transform"], cached["crs"])
    return StreamingResponse(
        io.BytesIO(tiff_bytes), media_type="image/tiff",
        headers={"Content-Disposition": f'attachment; filename="{index}_{boundary_id[:8]}.tif"'},
    )


@app.get("/api/export/shp/plots")
async def export_plots_shp(boundary_id: str):
    conn = get_db()
    rows = conn.execute(
        "SELECT plot_id, row_idx, col_idx, geojson, area_m2 FROM plots WHERE boundary_id = ? ORDER BY plot_id",
        (boundary_id,),
    ).fetchall()
    conn.close()
    if not rows:
        raise HTTPException(400, "No plots found — call /api/plots/generate-grid first.")

    features = [
        {"geometry": json.loads(r["geojson"]), "plot_id": r["plot_id"],
         "row": r["row_idx"], "col": r["col_idx"], "area_m2": r["area_m2"]}
        for r in rows
    ]
    zip_bytes = exports_mod.geometries_to_shapefile_zip_bytes(features, layer_name="plots")
    return StreamingResponse(
        io.BytesIO(zip_bytes), media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="plots_{boundary_id[:8]}.zip"'},
    )


@app.get("/api/export/shp/zones")
async def export_zones_shp(boundary_id: str, index: str, n_zones: int = 3):
    cached = _RESULT_CACHE.get(f"{boundary_id}:{index}:zones:{n_zones}")
    if not cached:
        raise HTTPException(400, "No zones cached — call /api/zones/classify first.")
    features = [
        {"geometry": z["geometry"], "zone": z["zone"], "pixel_count": z["pixel_count"],
         "rate": z["suggested_rate"]}
        for z in cached["zones"]
    ]
    zip_bytes = exports_mod.geometries_to_shapefile_zip_bytes(features, layer_name="zones")
    return StreamingResponse(
        io.BytesIO(zip_bytes), media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="zones_{index}_{boundary_id[:8]}.zip"'},
    )


@app.get("/api/export/excel/plots")
async def export_plots_excel(boundary_id: str, index: str):
    """Re-runs the same zonal-stats logic as /api/plots/zonal-stats and returns it as .xlsx."""
    resp = await plots_zonal_stats(boundary_id=boundary_id, index=index)
    plot_stats = resp["plots"]
    xlsx_bytes = exports_mod.plot_stats_to_excel_bytes(plot_stats, sheet_title=f"{index.upper()} Plot Stats")
    return StreamingResponse(
        io.BytesIO(xlsx_bytes),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="plot_stats_{index}_{boundary_id[:8]}.xlsx"'},
    )


@app.get("/api/export/pdf")
async def export_pdf(boundary_id: str, index: str):
    cached = _RESULT_CACHE.get(f"{boundary_id}:{index}")
    if not cached:
        raise HTTPException(400, f"No processed '{index}' result cached — process it first.")

    conn = get_db()
    b_row = conn.execute("SELECT name FROM boundaries WHERE id = ?", (boundary_id,)).fetchone()
    plot_count = conn.execute("SELECT COUNT(*) c FROM plots WHERE boundary_id = ?", (boundary_id,)).fetchone()["c"]
    conn.close()
    farm_name = b_row["name"] if b_row else "Unnamed Farm"

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
async def upload_drone_image(file: UploadFile = File(...)):
    """Upload a stitched drone orthomosaic GeoTIFF (Pix4D/Agisoft/DroneDeploy/
    OpenDroneMap output). Returns metadata so the frontend can confirm band
    mapping before running indices or plant counts on it."""
    if not file.filename.lower().endswith((".tif", ".tiff")):
        raise HTTPException(400, "Please upload a GeoTIFF (.tif/.tiff) orthomosaic")

    file_id = str(uuid.uuid4())
    dest = UPLOAD_DIR / f"{file_id}.tif"
    with open(dest, "wb") as f:
        f.write(await file.read())

    try:
        meta = drone_ingest.inspect_orthomosaic(str(dest))
        preview = drone_ingest.generate_preview(str(dest))
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(400, f"Could not read file as GeoTIFF: {e}")

    return {"drone_file_id": file_id, **meta, **preview}


@app.post("/api/drone/plant-count")
async def drone_plant_count(drone_file_id: str = Form(...), boundary_id: str = Form(None),
                             min_plant_area_px: int = Form(15)):
    """
    Runs blob/watershed-based plant detection on an uploaded drone orthomosaic.
    Requires centimeter-scale imagery — will not work meaningfully on satellite data.
    See plant_count.py docstring for accuracy caveats (baseline method, not a
    trained model — best on widely-spaced row/nursery crops).
    """
    file_path = UPLOAD_DIR / f"{drone_file_id}.tif"
    if not file_path.exists():
        raise HTTPException(404, "Uploaded drone file not found — upload it via /api/drone/upload first.")

    boundary_geojson = None
    if boundary_id:
        conn = get_db()
        row = conn.execute("SELECT geojson FROM boundaries WHERE id = ?", (boundary_id,)).fetchone()
        conn.close()
        if row:
            boundary_geojson = json.loads(row["geojson"])

    rgb, meta = drone_ingest.load_rgb_for_plant_count(str(file_path), boundary_geojson)
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
async def drone_plant_count_per_plot(drone_file_id: str = Form(...), boundary_id: str = Form(...),
                                      min_plant_area_px: int = Form(15)):
    """
    Plant counts broken down per plot (Solvi's 'statistics for plant counts' as part
    of per-plot Zonal Statistics), rather than one field-wide total. Requires plots
    to already exist for this boundary (from /api/plots/generate-grid,
    /api/plots/extract-from-drone, or /api/plots/import).
    """
    file_path = UPLOAD_DIR / f"{drone_file_id}.tif"
    if not file_path.exists():
        raise HTTPException(404, "Uploaded drone file not found — upload it via /api/drone/upload first.")

    conn = get_db()
    boundary_row = conn.execute("SELECT geojson FROM boundaries WHERE id = ?", (boundary_id,)).fetchone()
    plot_rows = conn.execute(
        "SELECT plot_id, row_idx, col_idx, geojson FROM plots WHERE boundary_id = ? ORDER BY plot_id",
        (boundary_id,),
    ).fetchall()
    conn.close()
    if not boundary_row:
        raise HTTPException(404, "Boundary not found")
    if not plot_rows:
        raise HTTPException(400, "No plots found for this boundary — generate or extract plots first.")

    boundary_geojson = json.loads(boundary_row["geojson"])
    plots = [
        {"plot_id": r["plot_id"], "row": r["row_idx"], "col": r["col_idx"],
         "geometry": json.loads(r["geojson"])}
        for r in plot_rows
    ]

    rgb, meta = drone_ingest.load_rgb_for_plant_count(str(file_path), boundary_geojson)
    result = plant_count_mod.count_plants(rgb, min_plant_area_px=min_plant_area_px)
    geo_points = plant_count_mod.pixel_centroids_to_geojson_points(
        result["centroids_px"], result["areas_px"], meta["transform"], meta["crs"]
    )
    per_plot = plant_count_mod.aggregate_plants_per_plot(geo_points, plots)

    return {
        "boundary_id": boundary_id,
        "total_plant_count": result["count"],
        "plot_count": len(per_plot),
        "plots": per_plot,
    }


# ---------------------------------------------------------------------------
# Annotations & sharing
# ---------------------------------------------------------------------------

@app.post("/api/annotations")
async def create_annotation(boundary_id: str = Form(...), lat: float = Form(...),
                             lng: float = Form(...), comment: str = Form(...),
                             author: str = Form(None)):
    return annotations_mod.add_annotation(boundary_id, lat, lng, comment, author)


@app.get("/api/annotations/{boundary_id}")
async def get_annotations(boundary_id: str):
    return {"annotations": annotations_mod.list_annotations(boundary_id)}


@app.post("/api/share")
async def create_share(boundary_id: str = Form(...), index: str = Form(None)):
    token = annotations_mod.create_share_link(boundary_id, index)
    return {"token": token, "share_path": f"/api/share/{token}"}


@app.get("/api/share/{token}")
async def resolve_share(token: str):
    link = annotations_mod.resolve_share_link(token)
    if not link:
        raise HTTPException(404, "Share link not found or expired")

    boundary = await get_boundary(link["boundary_id"])
    result = {"boundary": boundary, "index": link["index_name"]}

    if link["index_name"]:
        cached = _RESULT_CACHE.get(f"{link['boundary_id']}:{link['index_name']}")
        if cached:
            result["overlay_png_base64"] = base64.b64encode(cached["png_bytes"]).decode()
            result["stats"] = cached["stats"]

    return result


@app.get("/api/health")
async def health():
    return {"status": "ok"}


app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")
