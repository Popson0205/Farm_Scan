# FarmScan — Open-Satellite & Drone Farm Health Mapping

A working scaffold covering the full feature set: automatic plot/boundary
extraction, vegetation/water indices, management zones + prescription export,
plant counts (drone-resolution), and share/collaborate exports — the open-data
equivalent of Solvi's core feature set.

## Feature map (vs. Solvi)

| Solvi feature | FarmScan equivalent | Data source needed |
|---|---|---|
| Plant Counts & Size Estimation (PlantAI™) | `/api/drone/plant-count` — blob/watershed detection + canopy area | **Drone** orthomosaic (cm-res) |
| Zonal Statistics & Plot Extraction | `/api/plots/generate-grid`, `/api/plots/zonal-stats`, `/api/plots/detect-boundary` | Satellite or drone |
| Plant Health Maps & Prescriptions | `/api/imagery/process*` (6 indices) + `/api/zones/classify` | Satellite or drone |
| Imagery Processing on Autopilot | `/api/imagery/search` (Sentinel-2 STAC) + `/api/drone/upload` | Satellite (auto) / drone (user stitches first) |
| Share and Collaborate | `/api/annotations`, `/api/share`, `/api/export/*` | — |

**Why two imagery paths?** Satellite (Sentinel-2, 10m/pixel, free, automatic,
whole-farm) is great for field-level vegetation/water indices and management
zones. It **cannot** resolve individual plants — a pixel covers ~100m².
Plant counts need drone-grade resolution (0.5–5cm/pixel), so that feature
consumes an already-stitched drone orthomosaic (output of Pix4D / Agisoft
Metashape / DroneDeploy / OpenDroneMap) uploaded as a GeoTIFF — FarmScan
doesn't do the raw-photo stitching itself, matching how Solvi also expects
processed imagery at that stage.

## How it works

```
Farm boundary (draw/upload) ──┬─→ Search Sentinel-2 (Planetary Computer STAC)
                               │       → Fetch + clip bands → Compute index
                               │       → Color overlay + zonal stats (cached)
                               │
                               ├─→ Auto-generate grid plots → per-plot stats
                               │   (reuses the cached index raster)
                               │
                               ├─→ Classify into N management zones (KMeans)
                               │   → polygonize → prescription export
                               │
                               └─→ Auto-detect field boundary from veg mask
                                   (contour detection on thresholded index)

Drone orthomosaic (upload) ──→ Plant blob detection (ExG + watershed)
                               → georeferenced plant points + canopy area

Exports: GeoTIFF, Shapefile (.zip), Excel (.xlsx), PDF report
Collaborate: point annotations, shareable read-only links
```

**Backend**: FastAPI + rasterio + shapely + scikit-learn (zones) + OpenCV/scikit-image (plot & plant detection) + SQLite (swap for PostGIS/Postgres later)
**Frontend**: Plain HTML/JS + Leaflet + Leaflet.draw (no build step needed)
**Imagery source**: [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/) — free Sentinel-2 L2A STAC catalog, no API key required.

## Running it

### 1. Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Live satellite search/processing needs outbound access to
`planetarycomputer.microsoft.com`. Without that (e.g. a sandboxed dev
environment), use the demo endpoints (`process-demo`) — they generate
spatially-realistic synthetic reflectance data so the entire pipeline
(boundary → index → plots → zones → exports) can be tested offline. All of
this was verified working end-to-end in demo mode during development,
including every export format and the plant counter (55/60 = 92% detected on
a synthetic test image with well-separated blobs).

### 2. Frontend

```bash
cd frontend
python3 -m http.server 5500
```

Visit `http://localhost:5500`. Talks to the backend at `http://127.0.0.1:8000`
(edit `API_BASE` in `index.html` if deployed elsewhere).

## API reference

### Boundaries & imagery
| Endpoint | Method | Purpose |
|---|---|---|
| `/api/boundary` | POST | Import a farm boundary (`name`, `geojson`) |
| `/api/boundary/{id}` | GET | Fetch one boundary |
| `/api/boundaries` | GET | List all boundaries |
| `/api/indices` | GET | List available indices + required bands |
| `/api/imagery/search` | GET | Search live Sentinel-2 scenes |
| `/api/imagery/process` | POST | Compute an index on a live scene |
| `/api/imagery/process-demo` | POST | Same, offline synthetic data |

### Plots & zones (new)
| Endpoint | Method | Purpose |
|---|---|---|
| `/api/plots/generate-grid` | POST | Auto-split boundary into a rows×cols plot grid (`rows`+`cols` or `target_plot_count`) |
| `/api/plots/{boundary_id}` | GET | List saved plots |
| `/api/plots/detect-boundary` | POST | Auto-detect field boundary candidates from a processed index (contour detection) |
| `/api/plots/zonal-stats` | POST | Per-plot mean/min/max/std/pixel_count for a processed index |
| `/api/zones/classify` | POST | KMeans-cluster an index into N management zones + polygons |

### Exports (new)
| Endpoint | Method | Purpose |
|---|---|---|
| `/api/export/geotiff` | GET | Download the processed index as a GeoTIFF |
| `/api/export/shp/plots` | GET | Download plot boundaries as a zipped Shapefile |
| `/api/export/shp/zones` | GET | Download management zones (prescription) as a zipped Shapefile |
| `/api/export/excel/plots` | GET | Download per-plot stats as a formatted .xlsx |
| `/api/export/pdf` | GET | Download a one-page PDF field report |

### Drone plant count (new)
| Endpoint | Method | Purpose |
|---|---|---|
| `/api/drone/upload` | POST | Upload a stitched drone orthomosaic GeoTIFF |
| `/api/drone/plant-count` | POST | Detect individual plants + canopy area |

### Collaboration (new)
| Endpoint | Method | Purpose |
|---|---|---|
| `/api/annotations` | POST | Add a point comment/note |
| `/api/annotations/{boundary_id}` | GET | List notes for a farm |
| `/api/share` | POST | Create a read-only share token for a boundary+index |
| `/api/share/{token}` | GET | Resolve a share link (returns boundary + cached overlay/stats) |

## Indices implemented (`backend/indices.py`)

| Index | Formula basis | Use |
|---|---|---|
| NDVI | (NIR−Red)/(NIR+Red) | General vegetation vigor/density |
| NDRE | (NIR−RedEdge)/(NIR+RedEdge) | Nitrogen/chlorophyll stress, less saturation in dense canopy |
| NDWI | (Green−NIR)/(Green+NIR) | Open water / surface moisture |
| NDMI | (NIR−SWIR1)/(NIR+SWIR1) | Crop/canopy water stress |
| SAVI | Soil-adjusted NDVI variant | Sparse/young crop cover |
| EVI | Corrects for haze + canopy background | High-biomass areas where NDVI saturates |

VARI (Solvi's RGB-only visible-light index) can be added the same way if
you want an index that works on plain RGB drone photos without NIR — it's
not yet in `INDEX_REGISTRY`, add it as `(green - red) / (green + red - blue)`.

## AI-based plot extraction (MobileSAM)

`/api/plots/extract-from-drone` supports two detection methods via the
`method` form field:

- **`vegetation`** (default) — classical excess-green vegetation mask +
  contour detection (`plots.py`). Fast, zero extra dependencies. Assumes
  plots are distinguishable from their surroundings by canopy greenness —
  works well for young/sparse trial plots on bare soil, poorly when plots
  are instead delineated by access roads/tracks cutting through otherwise-
  continuous vegetation or bare ground (common on real plantation imagery).
- **`sam`** — zero-shot segmentation using [MobileSAM](https://github.com/ChaoningZhang/MobileSAM),
  a lightweight distillation of Meta's Segment Anything Model. It has never
  been trained on farm imagery specifically, but its automatic mask
  generation mode segments an image into all its visually distinct regions
  — which, on plantation/field photos, tends to track individual plot
  boundaries (defined by roads, tone, and texture) far better than a fixed
  color rule, because it's responding to edge structure jointly rather than
  one hand-tuned threshold. Validated against real oblique plantation
  drone photos during development: it cleanly separated nearly every
  individual cleared block along the true access-road boundaries.

**What's vendored vs. what you need to install:** the MobileSAM package code
(`backend/vendor/mobile_sam/`) is included in this repo. Its ~39MB pretrained
weights are **not** committed to git — that's over GitHub's web-upload UI
limit and unnecessarily bloats git history for a binary. Fetch them once
after cloning:

```bash
python3 backend/scripts/download_sam_weights.py
# or: bash backend/scripts/download_sam_weights.sh
```

This pulls `mobile_sam.pt` from the upstream MobileSAM GitHub repo into
`backend/models/mobile_sam.pt` (safe to re-run — skips if already present).
Add this as a build/deploy step (e.g. a Railway build command) if your
platform doesn't persist `backend/models/` between deploys. You also need
`torch` and `timm` installed (`pip install -r requirements.txt` covers this;
comment those two lines out if you don't want the ~800MB of extra
dependencies torch brings in — the app runs fine with `vegetation` as the
only method).
`GET /api/plots/extraction-methods` reports which methods are actually
usable at runtime so the frontend can grey out `sam` if the weights/
dependencies aren't present, instead of failing after upload.

**Performance**: SAM inference is CPU-slow (roughly 1-2 minutes per image at
default settings on a modest CPU instance — the backend downsamples to a
768px-longest-side working copy internally to keep memory bounded and avoid
OOM on small instances). For production use with real orthomosaics, either
run on a GPU instance (seconds instead of minutes) or keep `points_per_side`
low. This is a genuine zero-shot use of a general-purpose pretrained model,
not a farm-specific trained one — for meaningfully better accuracy, the
natural next step is fine-tuning on a labeled field-boundary dataset. See
the "Public datasets for a trained model" section below.

### Public datasets for a trained model

If you want to move past zero-shot SAM to an actually-trained field-boundary
model, these are the most relevant public datasets/models as of writing:

| Resource | What it is |
|---|---|
| [AI4Boundaries](https://essd.copernicus.org/articles/15/317/2023/) | 7,831 labeled tiles, 1m aerial ortho + Sentinel-2, vector ground-truth parcel boundaries (EU) |
| [Fields of the World (FTW)](https://arxiv.org/abs/2409.16252) | Global field-boundary instance segmentation benchmark, built to generalize across countries |
| [Delineate Anything / FBIS-22M](https://arxiv.org/abs/2504.02534) | 672,909 patches, 22.9M field instance masks, 0.25–10m resolution, with released pretrained weights — worth checking first for a stronger zero-shot baseline than MobileSAM |
| [AI4SmallFarms](https://github.com/tim-ov/AI4SmallFarms) | 439,001 field polygons, Vietnam/Cambodia smallholder farms — closer to irregular real-world plot shapes than EU rectangles |
| [DeepGlobe Road Extraction](https://arxiv.org/abs/1805.06561) | 6,226 images, 0.5m/px, road masks, imagery from Thailand/Indonesia/India — closest visual domain match to tropical unpaved plantation roads, useful for the "roads define plot boundaries" approach as an alternative to region segmentation |
| [Descals et al. global oil-palm map](https://essd.copernicus.org/articles/13/1211/2021/) | Same crop as this deployment; DeepLabv3+ trained on Sentinel-1/2, 10m resolution — good for coarse "where is oil palm" classification, not plot-level boundaries |

Practical path: fine-tune Delineate Anything or an FTW checkpoint on a few
dozen manually labeled tiles from your own plantation imagery — transfer
learning from a model already trained on millions of field-boundary
instances needs far less labeled data than training from scratch.

## Image acquisition requirements for real deployment

Plot extraction and plant counts both require a **nadir (straight-down)
orthomosaic GeoTIFF** stitched from a full drone flight — not a single
oblique photo (e.g. a phone shot or an angled gimbal capture out the side of
a plane). Oblique imagery has no consistent ground-sample-distance, so area
and boundary calculations will be systematically wrong, worse toward the
horizon. Tell your users:

1. Fly a **nadir (camera pointing straight down)** grid pattern over the
   field, not a single oblique pass.
2. Target **2–5cm/pixel** ground resolution (lower altitude = finer detail,
   needed for plant counts especially).
3. Capture with **75%+ front overlap** and **65%+ side overlap** between
   consecutive photos — this is what the stitching software needs to
   reconstruct accurate geometry.
4. Stitch the raw photos into a single georeferenced orthomosaic using
   Pix4D, DroneDeploy, Agisoft Metashape, or the free/open-source
   [OpenDroneMap](https://www.opendronemap.org/).
5. Export/upload the result as a **GeoTIFF (.tif)** — this is what
   `/api/drone/upload` expects; a JPEG/PNG screenshot or a raw un-stitched
   photo will not carry the georeferencing the rest of the pipeline needs.

## Honest limitations / what's next

1. **Plant counting is a baseline, not a trained model.** `plant_count.py`
   uses classical vegetation segmentation + watershed splitting (ExG index +
   distance transform). It scored 92% on a synthetic well-separated test —
   good for widely-spaced row/nursery crops, will undercount dense
   closed-canopy fields (cereals) where plants visually merge. Production
   parity with Solvi's PlantAI needs a crop-specific trained detector
   (e.g. YOLO fine-tuned on your crop imagery).
2. **No cloud/shadow masking** on satellite imagery yet — apply the
   Sentinel-2 SCL band before computing indices for production accuracy.
3. **SQLite** is a placeholder — swap for PostGIS (Supabase supports this as
   an extension, and you already use Supabase for GeoEstate) once ready for
   multi-user/production scale.
4. **Grid-based plot extraction** covers the common trial-layout case
   (regular rows/cols), and automatic extraction now has two methods —
   classical vegetation-mask contour detection, and zero-shot AI
   segmentation (MobileSAM) for road-delineated plots (see "AI-based plot
   extraction" above). Manual editing (`/api/plots/{boundary_id}/edit`,
   `/add`, `/{plot_id}` DELETE) and SHP/KML import
   (`/api/plots/import`) are also now built. What's still missing is a
   *farm-specific trained* model — both current methods are either a fixed
   color rule or a general-purpose pretrained model with no farm-imagery
   fine-tuning; see the public-dataset table above for the path to close
   that gap.
5. **Elevation maps** (from Solvi's Plant Health module) aren't implemented —
   would need a DEM source (e.g. Copernicus DEM, also free) and a similar
   fetch+clip pipeline to what `stac_client.py` already does for Sentinel-2.
6. **Time series / alerting** — not yet built. Each processed result is
   cached in memory only (cleared on server restart); persisting results
   with timestamps per field would unlock trend charts and threshold alerts.
7. Area/pixel-size math uses a simple degrees→meters approximation centered
   on the boundary's latitude — fine for display, reproject to a local UTM
   zone for anything contractual.
