# FarmScan — Open-Satellite & Drone Farm Health Mapping

Open-data farm health mapping, built to turn imagery into field decisions:
cloud-masked vegetation/water indices, a season timeline with alerts, problem
areas ranked for scouting, plot performance ranking, management zones with a
variable-rate plan, drone plant counts, and share/collaborate exports.

> **Scope note:** automatic plot extraction (vegetation-mask and MobileSAM
> methods, and automatic field-boundary detection) has been removed. Plots come
> from a regular grid, from importing an existing SHP/KML layer, or from
> drawing/editing on the map. That removed the ~800 MB torch dependency and a
> 39 MB weights download, and avoids plot boundaries that look authoritative
> but were guessed by a model.

## Feature map (vs. Solvi)

| Solvi feature | FarmScan equivalent | Data source needed |
|---|---|---|
| Plant Counts & Size Estimation (PlantAI™) | `/api/drone/plant-count` — blob/watershed detection + canopy area | **Drone** orthomosaic (cm-res) |
| Zonal Statistics & Plots | `/api/plots/generate-grid`, `/api/plots/import`, `/api/plots/{id}/edit`, `/api/plots/zonal-stats`, `/api/plots/performance` | Satellite or drone |
| Season timeline, alerts & scouting | `/api/monitoring/history`, `/api/monitoring/problem-areas` | Satellite (Sentinel-2) |
| Plant Health Maps & Prescriptions | `/api/imagery/process*` (6 indices) + `/api/zones/classify` | Satellite or drone |
| Imagery Processing on Autopilot | `/api/imagery/search` (Sentinel-2 STAC) + `/api/drone/upload` | Satellite (auto) / drone (user stitches first) |
| Share and Collaborate | `/api/annotations`, `/api/share`, `/api/export/*` | — |
| Site suitability & planting plan | `/api/suitability/analyze`, `/api/soil/profile`, `/api/weather/climate`, `/api/weather/forecast` | HWSD v2.0 (local files) + Open-Meteo / NASA POWER (free APIs) |

**Why two imagery paths?** Satellite (Sentinel-2, 10m/pixel, free, automatic,
whole-farm) is great for field-level vegetation/water indices and management
zones. It **cannot** resolve individual plants — a pixel covers ~100m².
Plant counts need drone-grade resolution (0.5–5cm/pixel), so that feature
consumes an already-stitched drone orthomosaic (output of Pix4D / Agisoft
Metashape / DroneDeploy / OpenDroneMap) uploaded as a GeoTIFF — FarmScan
doesn't do the raw-photo stitching itself, matching how Solvi also expects
processed imagery at that stage.

## Site suitability: soil + weather -> crop decisions

```
boundary ──┬─→ HWSD v2.0 raster (local, ~1 km)  → soil mapping units under the farm
           │       → HWSD2_LAYERS (SQLite): sand/silt/clay, pH, organic C, CEC, EC, drainage ...
           │       → share- and depth-weighted topsoil (0-30 cm) / subsoil (30-100 cm)
           │
           ├─→ Open-Meteo ERA5 archive (fallback NASA POWER), 10 yrs daily, cached in SQLite
           │       → monthly normals, rainy season, per-year rain / temperature / dry spells
           │
           └─→ suitability.py + crop_requirements.json
                   → per-factor 0-1 scores → class by most limiting factor (S1/S2/S3/N)
                   → best planting windows (rain onset, cycle rain, dry spells, heat)
                   → "what limits it" + management advice
```

**HWSD is not a web API** -- FAO distributes it as files. FarmScan hosts them locally (offline,
no rate limits). One-time setup:

1. Download the HWSD v2.0 raster and database from the FAO soils portal (see the header of
   `backend/scripts/build_hwsd_sqlite.py` for the link).
2. Put `HWSD2.bil` (+ `.hdr`/`.prj`) in `backend/data/hwsd/`.
3. `python backend/scripts/build_hwsd_sqlite.py --mdb /path/to/HWSD2.mdb`
   (or `--layers-csv` for a CSV export of the `HWSD2_LAYERS` table).
4. `GET /api/data-status` should report `"ready": true`.

Environment: `HWSD2_RASTER`, `HWSD2_DB` (override paths), `OPEN_METEO_API_KEY` +
`OPEN_METEO_BASE_URL` (needed for **commercial** use of Open-Meteo; the free tier is
non-commercial), `CROP_REQUIREMENTS` (path to your own crop table).

**Read this before trusting a score**
- HWSD pixels are ~1 km (~100 ha). A normal farm falls inside one pixel, so you get the *regional
  soil association*, not your field. Use it to screen crops and plan; use a **lab soil test** for
  fertilizer and lime rates. Plots inside one farm will not differ in soil data.
- Reanalysis weather is ~9-25 km. Rainfall in particular is patchy; a local gauge beats it.
- `crop_requirements.json` holds starter thresholds from general FAO/EcoCrop-style literature.
  **Calibrate them with a local agronomist** (edit the file; no code change) before relying on
  rankings. Scores assume rainfed production.
- Satellite NDVI (the rest of FarmScan) is the reality check: if a field the model calls S1 shows
  persistent low NDVI, something not in this data (compaction, pests, management) is limiting it.

## How it works

```
Farm boundary (draw/upload) ──┬─→ Search Sentinel-2 (Planetary Computer STAC)
                               │       → Fetch + clip bands → Compute index
                               │       → Color overlay + zonal stats (cached)
                               │
                               ├─→ Every run saved as a dated observation
                               │   → season timeline + drop alerts
                               │
                               ├─→ Problem areas (patches below the field norm)
                               │   → ranked, with scouting notes on the map
                               │
                               ├─→ Grid / imported / hand-edited plots
                               │   → per-plot stats → performance ranking
                               │
                               └─→ Classify into N management zones (KMeans)
                                   → rate plan (agronomist's base rate)
                                   → prescription export

Drone orthomosaic (upload) ──→ Plant blob detection (ExG + watershed)
                               → georeferenced plant points + canopy area

Exports: GeoTIFF, Shapefile (.zip), Excel (.xlsx), PDF report
Collaborate: point annotations, shareable read-only links
```

**Backend**: FastAPI + rasterio + shapely + scikit-learn (zones) + OpenCV/scikit-image (plot & plant detection) + SQLite (swap for PostGIS/Postgres later)
**Frontend**: Plain HTML/JS + Leaflet + Leaflet.draw (no build step needed)
**Imagery source**: [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/) — free Sentinel-2 L2A STAC catalog, no API key required.

## On the GeoEstate platform

FarmScan is one of the GeoEstate products. It has no accounts of its own:

- **Sign-in** is GeoEstate ID (Geohub). People open FarmScan, sign in with their GeoEstate account, and work in their
  organization. Roles come from their Geohub role: owner/admin = **manager**, member = **agronomist**,
  collector/viewer (and GeoEstate staff) = **viewer** (read only).
- **The licence** is the organization's FarmScan entitlement in Geohub (sold and invoiced in the CRM). Its plan switches
  features on (Standard: monitoring, zones, suitability; Advanced adds drone, plant counts, prescriptions) and sets the
  **hectares** of farms the organization can monitor. Expired = read-only, suspended = closed.
- **Data** lives in the platform database (PostgreSQL + PostGIS), in the `farm` schema: farms, plots, the season
  timeline, notes, share links. Nothing is lost when the service redeploys.
- **Files** (drone orthomosaics, the soil dataset) live in the platform bucket under `farm/`.

### Many farms, field teams and scouts (with Geohub)

- **All farms**: every farm of the organization on one map, coloured by its latest NDVI and ranked weakest first
  (dropping since the previous image first), with open scouting tasks (`GET /api/portfolio`).
- **Farms from the field**: a field team walks each farm's boundary with a Geohub form started from the **Farm
  enumeration** template (farmer, phone, community, crops, planting date, tenure, water). "From Geohub" brings every
  submitted or approved record in as a farm with its farmer's details; run it again as more come in
  (`POST /api/enumeration/import`). The plan's hectares apply.
- **Scouting**: a manager picks the organization's Geohub form started from the **Farm scouting** template once.
  "Send problem areas to scouts" makes one task per problem area, each a link that opens that form in the Geohub
  collector with the farm, the task and what the satellite saw filled in and the place marked; share them on WhatsApp.
  "Check scouts' answers" reads the records back by task and shows the cause, severity and photos count per area.

### The season through the clouds (Sentinel-1 radar) and planting/harvest dates

- **Build the season from satellite** (`POST /api/season/build`, a background job): NDVI from *every* Sentinel-2
  image covering the farm over the dates (each pixel cloud-masked; images under 60% clear are kept but not used), and
  **Sentinel-1 radar** for every pass — RVI = 4·VH/(VV+VH) and VH backscatter — which cloud does not stop. From May to
  September clouds often hide Nigerian farms from optical satellites for weeks; radar fills those gaps.
- **The season reading** (`GET /api/season`, `backend/season.py`): green-up (20% of the season's amplitude, rising),
  peak, and harvest or dry-down (50%, falling), the amplitude-threshold method of TIMESAT / MODIS land-cover dynamics,
  each with its uncertainty in days (half the gap between the images around it). It uses optical NDVI unless clouds
  left a gap around green-up that radar fills better. **Sowing** is given as a range before green-up (a crop shows from
  space 1–4 weeks after sowing). Where a farm came from Geohub with a planting date, it is checked against this.
- No clear crop cycle (amplitude under 0.15 NDVI / 0.10 RVI), too few images, already green at the first image, or not
  yet harvested at the last are said as such rather than guessed.

### Deploying (Railway)

The service is `backend/` (Railpack, `uvicorn main:app`), which also serves the web app. Variables:

| Variable | Value |
|---|---|
| `DATABASE_URL` | the platform database (Geohub's) |
| `APP_URL` | FarmScan's own address, e.g. `https://farm.geoestate.ng` |
| `GEOHUB_URL` / `GEOHUB_API` | Geohub's address and its API (`…/api`) |
| `ID_SECRET` | the secret Geohub's `ID_CLIENTS` lists for `farm` |
| `S3_ENDPOINT`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`, `S3_BUCKET`, `S3_REGION` | the platform bucket (same as Geohub's) |
| `DEMO_TOOLS` | leave unset; `true` only on a demonstration copy (shows "practice with sample data") |
| `OPEN_METEO_API_KEY` | optional: Open-Meteo's commercial key |

In Geohub: add `{"id": "farm", "redirect_uris": ["<APP_URL>/api/auth/callback"], "secret": "<ID_SECRET>"}` to
`ID_CLIENTS`, and `farm=<APP_URL>` to `PRODUCT_URLS` (the app switcher).

**Soil data:** upload `HWSD2.bil`, `HWSD2.hdr`, `HWSD2.prj` and `hwsd2.sqlite` once to the bucket under
`farm/data/hwsd/` (build the sqlite with `backend/scripts/build_hwsd_sqlite.py`). FarmScan fetches them at start-up.

### Running it locally

```bash
cd backend
pip install -r requirements-dev.txt
DATABASE_URL=postgresql://… GEOHUB_URL=… GEOHUB_API=… ID_SECRET=… uvicorn main:app --reload --port 8000
```

Live satellite search/processing needs outbound access to `planetarycomputer.microsoft.com`. With `DEMO_TOOLS=true`,
synthetic sample data exercises the whole pipeline offline; it is labelled as sample data and kept apart from real
images in the timeline.

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

### Plots & zones
| Endpoint | Method | Purpose |
|---|---|---|
| `/api/plots/generate-grid` | POST | Split boundary into a rows×cols plot grid (`rows`+`cols` or `target_plot_count`) |
| `/api/plots/import` | POST | Import plots from a zipped Shapefile or KML |
| `/api/plots/{boundary_id}` | GET | List saved plots |
| `/api/plots/{boundary_id}/edit` · `/add` · `DELETE /{plot_id}` | POST/DELETE | Edit, add, delete individual plots |
| `/api/plots/zonal-stats` | POST | Per-plot mean/min/max/std/pixel_count for a processed index |
| `/api/plots/performance` | POST | Rank plots vs the field baseline (z-score, flags, uniformity, headroom) |
| `/api/zones/classify` | POST | KMeans-cluster an index into N zones. Add `base_rate` (+ `unit`, `strategy`, `spread`) for a rate plan |

### Field monitoring
| Endpoint | Method | Purpose |
|---|---|---|
| `/api/monitoring/history` | GET | Season timeline for a field+index: change vs previous/baseline, drop alerts, cloudy images excluded |
| `/api/monitoring/problem-areas` | POST | Patches clearly below the field norm, ranked by impact; `create_scouting_notes=true` drops notes on the map |
| `/api/imagery/process-demo` | POST | Accepts optional `obs_date` (YYYY-MM-DD) to simulate a season offline |

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

## Using FarmScan for production decisions

**1. Watch the season, not a snapshot.** Every index run is saved as a dated
observation (field mean, spread, share of cloud-free pixels). The timeline flags a
field that drops ≥0.05 vs the previous image or ≥0.07 vs its recent baseline.
Images with under 60% clear pixels are kept but excluded from alerts, so a cloud
can't trigger a false alarm. Natural senescence and harvest also lower vegetation
indices — an alert is a prompt to look, not a diagnosis.

**2. Know where to walk first.** *Find problem areas* flags contiguous patches
more than 1.5 robust standard deviations below the field median (sensitivity is
adjustable), ranks them by area × shortfall, and tells you what to check first for
the index you ran (e.g. NDRE → nitrogen/chlorophyll; NDMI → water stress). It is
relative to the field's own distribution, so it needs no per-crop thresholds.
Ticking *Drop scouting notes* places a note at each patch; re-running replaces
earlier auto-generated notes and never touches notes written by people.

**3. Find weak plots in a trial or multi-plot field.** *Rank plots* orders plots
against the field baseline and reports uniformity (CV) and the headroom between the
median plot and the top quartile.

**4. Turn zones into a rate plan.** Enter your flat-rate recommendation (e.g. 90 kg
N/ha) and choose a strategy: *more on weak zones* when the shortfall is
nutrient-limited, *more on strong zones* when weak areas are limited by something
inputs won't fix (waterlogging, compaction, shallow soil). The plan shows rate, area
and product per zone and the total vs flat-rate. It redistributes input — any saving
or yield gain should be validated with a flat-rate strip before you rely on it.

## Accuracy fixes (satellite pipeline)

These change the numbers, so re-run any saved results:

- **Cloud/shadow masking** with the Sentinel-2 Scene Classification Layer (cloud,
  cirrus, shadow, saturated, snow, no-data are excluded) — important in the Nigerian
  rainy season, when low scene-level cloud cover can still mean cloud over the field.
- **Reflectance offset**: scenes processed with baseline ≥ 04.00 carry a +1000 DN
  offset that was previously ignored, which biased every index (NDVI read too low).
  The offset is read from the STAC asset metadata with a baseline-based fallback.
- **Mixed-resolution bands**: NDRE and NDMI combine 10 m and 20 m bands, which
  previously raised a shape-mismatch error on live data. All bands are now warped to
  one 10 m grid.
- **Zone pixel counts** counted polygon fragments instead of pixels; fixed (zone
  areas and rate totals depend on it).
- Annotation text is now HTML-escaped in the UI (it was injected as raw HTML).

> The live-imagery changes were verified against synthetic GeoTIFFs (UTM, 10 m and
> 20 m bands, SCL cloud/shadow blocks) because the build environment can't reach
> Planetary Computer. Do one live check on first deploy: run NDVI for a known
> vegetated field and compare its mean against the same scene in Sentinel Hub EO
> Browser.

## Tests

```bash
cd backend
pip install -r requirements-dev.txt
python3 -m pytest
```

Covers problem-area detection, trend alerts, plot ranking, rate-plan maths, STAC band
handling (offset, cloud mask, mixed resolution), the full API flow on sample data, and the platform rules
(organizations, roles, plan features, hectares, files, sign-in). Needs PostgreSQL with PostGIS: `DATABASE_URL`
(default `postgresql://hub:hub@localhost:5432/farm_test`).

## Image acquisition requirements for real deployment

Plant counts require a **nadir (straight-down)
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
2. **Cloud masking is per-scene.** Masked pixels become gaps rather than being
   filled from other dates. In heavy-cloud periods a multi-scene composite
   (median of the clear pixels across several dates) would give fuller coverage.
3. **SQLite** is a placeholder — swap for PostGIS (Supabase supports this as
   an extension, and you already use Supabase for GeoEstate) once ready for
   multi-user/production scale.
4. **Plots are not auto-detected.** Plot layouts come from a regular grid, an
   imported SHP/KML, or manual drawing/editing. For plantation-style layouts defined by
   roads and tracks, digitize or import the blocks once; they then persist for every
   later image.
5. **Elevation maps** (from Solvi's Plant Health module) aren't implemented —
   would need a DEM source (e.g. Copernicus DEM, also free) and a similar
   fetch+clip pipeline to what `stac_client.py` already does for Sentinel-2.
6. **Timeline is field-level.** Observations (field mean/spread) persist in SQLite,
   but the raster itself is cached in memory only, so problem areas, zones and
   per-plot stats need the image re-run after a server restart. Per-plot history and
   push alerts (SMS/WhatsApp/email) are not built yet.
   Alert thresholds are simple fixed rules — tune them against your own crops.
7. Area/pixel-size math uses a simple degrees→meters approximation centered
   on the boundary's latitude — fine for display, reproject to a local UTM
   zone for anything contractual.
