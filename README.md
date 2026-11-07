# FarmScan — satellite farm monitoring (GeoEstate)

Farm health from free satellite data, built to turn imagery into field decisions: cloud-masked vegetation and water
indices from Sentinel-2, a season timeline with alerts, radar from Sentinel-1 that sees through rainy-season cloud,
planting and harvest dates, problem areas sent to scouts in the Geohub field app, management zones with a
variable-rate plan, crop suitability from soil and climate, weekly alerts, and numbered verification reports.

> **Scope:** FarmScan works from 10 m satellite images. Tools that need drone-grade imagery (2–5 cm) — plant
> counts, drone orthomosaics, per-plot statistics and automatic plot extraction — have been removed, so nothing on
> the page promises a precision the data cannot give.

## Site suitability: soil + weather -> crop decisions

```
boundary ──┬─→ HWSD v2.0 raster (local, ~1 km)  → soil mapping units under the farm
           │       → HWSD2_LAYERS (SQLite): sand/silt/clay, pH, organic C, CEC, EC, drainage ...
           │       → share- and depth-weighted topsoil (0-30 cm) / subsoil (30-100 cm)
           │
           ├─→ Open-Meteo ERA5 archive (fallback NASA POWER), 10 yrs daily, cached in the database
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
  fertilizer and lime rates. Fields inside one farm will not differ in soil data.
- Reanalysis weather is ~9-25 km. Rainfall in particular is patchy; a local gauge beats it.
- `crop_requirements.json` holds starter thresholds from general FAO/EcoCrop-style literature.
  **Calibrate them with a local agronomist** (edit the file; no code change) before relying on
  rankings. Scores assume rainfed production.
- Satellite NDVI (the rest of FarmScan) is the reality check: if a field the model calls S1 shows
  persistent low NDVI, something not in this data (compaction, pests, management) is limiting it.

## How it works

```
Farm (drawn, uploaded, or walked in Geohub) ──┬─→ Search Sentinel-2 (Planetary Computer STAC)
                                              │       → fetch + clip bands, cloud-mask per pixel → index
                                              │       → colour overlay + field stats (cached)
                                              │
                                              ├─→ Every image saved as a dated observation
                                              │   + Sentinel-1 radar (RVI, VH) through cloud
                                              │   → season timeline, drop alerts, green-up / peak / harvest
                                              │
                                              ├─→ Problem areas (patches below the field norm)
                                              │   → scouting tasks → Geohub collector → findings back
                                              │
                                              └─→ Management zones (KMeans) → rate plan → prescription export

Weather (Open-Meteo) + fires (NASA FIRMS) + timeline → weekly alerts (in FarmScan and by email)
Exports: GeoTIFF, Shapefile (.zip), PDF; verification reports (numbered, checkable)
```

**Backend**: FastAPI + rasterio + shapely + scikit-learn (zones) + PostgreSQL/PostGIS (platform database)
**Frontend**: Plain HTML/JS + Leaflet + Leaflet.draw (no build step needed)
**Imagery source**: [Microsoft Planetary Computer](https://planetarycomputer.microsoft.com/) — free Sentinel-2 L2A and Sentinel-1 RTC, no API key required.

## On the GeoEstate platform

FarmScan is one of the GeoEstate products. It has no accounts of its own:

- **Sign-in** is GeoEstate ID (Geohub). People open FarmScan, sign in with their GeoEstate account, and work in their
  organization. Roles come from their Geohub role: owner/admin = **manager**, member = **agronomist**,
  collector/viewer (and GeoEstate staff) = **viewer** (read only).
- **The licence** is the organization's FarmScan entitlement in Geohub (sold and invoiced in the CRM). Its plan switches
  features on (Standard: monitoring, zones, suitability, alerts; Advanced adds prescriptions and verification reports) and sets the
  **hectares** of farms the organization can monitor. Expired = read-only, suspended = closed.
- **Data** lives in the platform database (PostgreSQL + PostGIS), in the `farm` schema: farms, the season
  timeline, notes, share links. Nothing is lost when the service redeploys.
- **Files** (the soil dataset) live in the platform bucket under `farm/`.

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

### Weekly alerts and verification reports

- **Weekly alerts** (`GET /api/alerts`, plan feature `alerts`): for every farm, a dry spell in the rainy season (10+
  days without ≥ 1 mm and little forecast), heavy rain (≥ 50 mm) or heat (≥ 38 °C) in the next week (Open-Meteo),
  satellite fire detections on or within 500 m of the farm in the last 7 days (NASA FIRMS VIIRS on NOAA-20 and NOAA-21 —
  Suomi NPP stopped on 2 Nov 2026 — needs `FIRMS_MAP_KEY`, free from the "Get MAP Key" button at
  https://firms.modaps.eosdis.nasa.gov/api/area/), and crop drops in the farm's timeline in the last 21 days. People who
  ask for them get them by email once a week (an hourly check in the service; one email per organization per week,
  only while its licence includes alerts and is current). Email goes through `BREVO_API_KEY` (or `RESEND_API_KEY`, `SMTP_*`), from
  `MAIL_FROM` — the same as Geohub's.
- **Verification report** (`POST /api/reports/verification`, plan feature `verification`): a numbered PDF with the
  farm's measured (and declared) area and outline, its crop seasons from the timeline (last three years), rainfall
  against the 10-year average, and the title status from GeoEstate Land Registry where `LAND_API` and
  `LAND_PLATFORM_KEY` are set (as for LandCheck). Kept as issued; anyone checks it at `/verify.html?code=FSV-…`
  (which shows only the farm's name, area and place, not the farmer's details).

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
| `BREVO_API_KEY` (or `RESEND_API_KEY`, or `SMTP_*`), `MAIL_FROM`, `MAIL_REPLY_TO` | weekly alert emails: copy Geohub's values (the same email service) |
| `FIRMS_MAP_KEY` | optional: NASA FIRMS key for fire alerts |
| `LAND_API`, `LAND_PLATFORM_KEY` | optional: title status in verification reports (Land Registry's `PLATFORM_KEY`) |

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

### Zones
| Endpoint | Method | Purpose |
|---|---|---|
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
| `/api/export/shp/zones` | GET | Download management zones (prescription) as a zipped Shapefile |
| `/api/export/pdf` | GET | Download a one-page PDF field report |

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

**3. Turn zones into a rate plan.** Enter your flat-rate recommendation (e.g. 90 kg
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

Covers problem-area detection, trend alerts, rate-plan maths, STAC band
handling (offset, cloud mask, mixed resolution), the full API flow on sample data, and the platform rules
(organizations, roles, plan features, hectares, files, sign-in). Needs PostgreSQL with PostGIS: `DATABASE_URL`
(default `postgresql://hub:hub@localhost:5432/farm_test`).

## Honest limitations / what's next

1. **10 m pixels.** Sentinel-2 cannot see individual plants or small plots; a farm under about 0.5 ha is only a few
   dozen pixels, so its numbers are rough. Use FarmScan for field- and farm-level decisions.
2. **Cloud masking is per image.** Masked pixels become gaps rather than being filled from other dates; the season
   reading uses radar where clouds leave gaps.
3. **The index raster is cached in memory.** Problem areas, zones and exports need the image re-run after a restart
   (the timeline itself is kept in the database).
4. **Elevation maps** aren't implemented — they would need a DEM (e.g. Copernicus DEM, free) and the same fetch+clip
   pipeline.
5. **Alert thresholds are fixed rules** — tune them against your own crops and areas.
