"""
Plot extraction: splits a farm/field boundary into individual trial plots,
either automatically (regular grid — the common case for row-crop / field-trial
layouts, matching Solvi's "Automatic Plot extraction") or by detecting field
edges directly from imagery (for irregular farm boundaries where none exists yet).
"""
import numpy as np
import cv2
from shapely.geometry import Polygon, box, shape, mapping
from shapely.ops import unary_union, transform as shapely_transform
from rasterio.features import shapes as rio_shapes
from rasterio.transform import from_bounds, xy
from rasterio.warp import transform_geom


# ---------------------------------------------------------------------------
# 1. Automatic grid plot generation (regular trial layout)
# ---------------------------------------------------------------------------

def generate_grid_plots(boundary_geojson: dict, rows: int, cols: int,
                         row_gap_m: float = 0.0, col_gap_m: float = 0.0) -> list[dict]:
    """
    Splits the boundary's bounding envelope into a rows x cols grid of plots,
    each clipped to the boundary polygon (so partial/edge plots follow the true
    field shape). Optional gaps simulate alleyways/buffers between plots.

    Returns a list of {plot_id, row, col, geometry (GeoJSON), area_m2}.
    Coordinates are assumed EPSG:4326; gaps are converted from meters using a
    local degree-per-meter approximation centered on the boundary.
    """
    geom = shape(boundary_geojson)
    minx, miny, maxx, maxy = geom.bounds
    lat = geom.centroid.y

    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * np.cos(np.radians(lat))
    gap_deg_x = col_gap_m / m_per_deg_lon
    gap_deg_y = row_gap_m / m_per_deg_lat

    cell_w = (maxx - minx) / cols
    cell_h = (maxy - miny) / rows

    plots = []
    plot_id = 0
    for r in range(rows):
        for c in range(cols):
            cx0 = minx + c * cell_w + gap_deg_x / 2
            cx1 = minx + (c + 1) * cell_w - gap_deg_x / 2
            cy0 = miny + r * cell_h + gap_deg_y / 2
            cy1 = miny + (r + 1) * cell_h - gap_deg_y / 2
            if cx1 <= cx0 or cy1 <= cy0:
                continue
            cell = box(cx0, cy0, cx1, cy1)
            clipped = cell.intersection(geom)
            if clipped.is_empty or clipped.area <= 0:
                continue

            # area in m^2 via the same local approximation
            area_m2 = clipped.area * m_per_deg_lat * m_per_deg_lon

            plots.append({
                "plot_id": plot_id,
                "row": r,
                "col": c,
                "geometry": mapping(clipped),
                "area_m2": round(area_m2, 1),
            })
            plot_id += 1

    return plots


def estimate_grid_from_plot_count(boundary_geojson: dict, target_plot_count: int) -> tuple[int, int]:
    """Suggests rows/cols that roughly match a target number of plots and the
    boundary's aspect ratio (used when the user just says 'give me ~40 plots')."""
    geom = shape(boundary_geojson)
    minx, miny, maxx, maxy = geom.bounds
    aspect = (maxx - minx) / max(maxy - miny, 1e-9)
    cols = max(1, round((target_plot_count * aspect) ** 0.5))
    rows = max(1, round(target_plot_count / cols))
    return rows, cols


# ---------------------------------------------------------------------------
# 2. Automatic field boundary detection from imagery
# ---------------------------------------------------------------------------

def detect_field_boundary(veg_mask: np.ndarray, transform, min_area_px: int = 500) -> list[dict]:
    """
    Detects candidate field boundaries from a binary vegetation/cropped-area mask
    (e.g. NDVI > threshold) by finding external contours, smoothing them, and
    converting pixel-space polygons to geographic coordinates via the raster
    transform. Useful when a user has imagery but no digitized boundary yet —
    they get candidate polygons to accept/edit rather than hand-tracing.

    veg_mask: 2D uint8 array, 0/1 or 0/255
    transform: rasterio Affine transform for the source raster
    Returns a list of {geometry (GeoJSON polygon), area_px, confidence}.
    """
    mask = (veg_mask > 0).astype(np.uint8) * 255

    # morphological cleanup: close small gaps, remove speckle noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        area_px = cv2.contourArea(cnt)
        if area_px < min_area_px:
            continue

        # simplify contour to reduce vertex count, then convert to geo coords
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if len(approx) < 3:
            continue

        coords = []
        for pt in approx.reshape(-1, 2):
            px, py = float(pt[0]), float(pt[1])
            gx, gy = xy(transform, py, px)  # row=py, col=px
            coords.append((gx, gy))
        coords.append(coords[0])  # close ring

        poly = Polygon(coords)
        if not poly.is_valid or poly.area == 0:
            poly = poly.buffer(0)
            if poly.is_empty:
                continue

        # rough confidence heuristic: how rectangular/compact the shape is
        # (farm fields tend toward compact/regular shapes vs. noise blobs)
        rect_area = cv2.minAreaRect(cnt)
        (rw, rh) = rect_area[1]
        rectangularity = area_px / max(rw * rh, 1e-6) if rw and rh else 0
        confidence = round(min(1.0, rectangularity), 2)

        # area in real-world units: polygon coords are in the raster's CRS units
        # (meters, for a UTM-projected drone orthomosaic — the normal case)
        area_m2 = float(poly.area)

        candidates.append({
            "geometry": mapping(poly),
            "area_px": int(area_px),
            "area_m2": round(area_m2, 2),
            "confidence": confidence,
            "_centroid": (poly.centroid.x, poly.centroid.y),
        })

    candidates.sort(key=lambda c: c["area_px"], reverse=True)
    return candidates


def veg_mask_from_ndvi(ndvi_array: np.ndarray, threshold: float = 0.3) -> np.ndarray:
    """Simple threshold to separate vegetated/cropped area from bare soil/background,
    used as input to detect_field_boundary()."""
    mask = np.where(np.nan_to_num(ndvi_array, nan=-1) > threshold, 1, 0).astype(np.uint8)
    return mask


# ---------------------------------------------------------------------------
# 3. Automatic plot extraction from an uploaded drone image
# ---------------------------------------------------------------------------

def extract_plots_from_drone_mask(veg_mask: np.ndarray, transform, crs, min_area_px: int = 300,
                                   row_gap_tolerance: float = 0.6) -> list[dict]:
    """
    Automatic plot extraction (Solvi-style): finds individual trial plots directly
    from an uploaded drone orthomosaic's vegetation mask — each distinct planted
    area/canopy cluster is detected as its own plot via contour detection, rather
    than imposing an arbitrary grid over the farm boundary.

    This reuses detect_field_boundary()'s contour-detection core (same CV pipeline),
    reprojects the resulting polygons from the drone's native CRS (typically a
    projected UTM CRS in meters) into EPSG:4326 to match every other geometry in
    the app (boundaries, satellite indices, the Leaflet map), then assigns each
    plot a (row, col) position by clustering plot centroids spatially.

    veg_mask: 2D uint8 array from excess_green_mask() (drone RGB) or veg_mask_from_ndvi()
              (if the drone sensor has NIR)
    transform: rasterio Affine transform for the source drone raster
    crs: the drone raster's native CRS (e.g. from rasterio's src.crs) — required to
         reproject detected plots into EPSG:4326
    min_area_px: minimum blob size to count as a plot (filters out speckle/weeds);
                 much smaller default than detect_field_boundary since individual
                 plots are far smaller than a whole field
    Returns a list of {plot_id, row, col, geometry (GeoJSON, EPSG:4326), area_m2, confidence}.
    """
    candidates = detect_field_boundary(veg_mask, transform, min_area_px=min_area_px)

    # reproject each detected polygon from the drone's native CRS into EPSG:4326
    # (area_m2 was computed in the drone's projected CRS above, which is accurate;
    # keep that value rather than recomputing from degree-based coordinates)
    for c in candidates:
        geom_4326 = transform_geom(crs, "EPSG:4326", c["geometry"])
        poly_4326 = shape(geom_4326)
        c["geometry"] = mapping(poly_4326)
        c["_centroid"] = (poly_4326.centroid.x, poly_4326.centroid.y)

    return _assign_grid_positions(candidates, row_gap_tolerance=row_gap_tolerance)


def _assign_grid_positions(candidates: list[dict], row_gap_tolerance: float = 0.6) -> list[dict]:
    """
    Groups detected plot polygons into row/col indices based on their centroids,
    so irregularly-shaped auto-detected plots still get a sensible row/col label
    (for display and CSV/shapefile ordering) without assuming a perfect grid.

    Rows are formed by sorting centroids by y and grouping ones whose y-gap is
    less than row_gap_tolerance * median plot extent (converted to degrees, since
    centroids are already in EPSG:4326 by the time this runs) — i.e. a new row
    starts once there's a real vertical jump. Within each row, plots are ordered
    left-to-right.
    """
    if not candidates:
        return []

    # median plot "diameter" from area (m^2 -> meters -> degrees, so it's on the
    # same scale as the EPSG:4326 centroid coordinates being compared)
    areas = [c["area_m2"] for c in candidates if c["area_m2"] > 0]
    median_extent_m = (np.median(areas) ** 0.5) if areas else 1.0
    median_extent_deg = median_extent_m / 111320.0
    row_break_dist = max(median_extent_deg * row_gap_tolerance, 1e-9)

    # sort by y descending (north to south) to assign rows top-to-bottom
    ordered = sorted(candidates, key=lambda c: -c["_centroid"][1])

    rows = []
    current_row = [ordered[0]]
    for c in ordered[1:]:
        prev_y = current_row[-1]["_centroid"][1]
        if abs(prev_y - c["_centroid"][1]) > row_break_dist:
            rows.append(current_row)
            current_row = [c]
        else:
            current_row.append(c)
    rows.append(current_row)

    plot_list = []
    plot_id = 0
    for row_idx, row in enumerate(rows):
        row_sorted = sorted(row, key=lambda c: c["_centroid"][0])  # left to right
        for col_idx, c in enumerate(row_sorted):
            plot_list.append({
                "plot_id": plot_id,
                "row": row_idx,
                "col": col_idx,
                "geometry": c["geometry"],
                "area_m2": c["area_m2"],
                "confidence": c["confidence"],
            })
            plot_id += 1

    return plot_list
