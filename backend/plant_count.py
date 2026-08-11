"""
Plant-level counting and canopy size estimation.

IMPORTANT: This requires centimeter-scale imagery (drone RGB orthomosaics,
typically 0.5-5cm/pixel). Satellite imagery like Sentinel-2 (10m/pixel) cannot
resolve individual plants — one pixel covers ~100 m^2. Feed this a drone-stitched
GeoTIFF (from Pix4D/Agisoft/DroneDeploy/OpenDroneMap), not a satellite scene.

Approach: this is a classical (non-ML) blob-detection method — vegetation
segmentation (excess-green index) + local-maxima counting + watershed splitting
for touching canopies. It's a reasonable baseline for widely-spaced row crops
(vegetables, young trees, nursery stock) but will under/over-count for dense
closed-canopy fields (e.g. mature cereals) where individual plants visually
merge — Solvi's production PlantAI is a trained deep model and will outperform
this on those cases. Good starting point; swap in a trained detector
(e.g. YOLO fine-tuned on your crop) for production-grade accuracy.
"""
import numpy as np
import cv2
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from shapely.geometry import shape, Point, mapping
from rasterio.transform import xy


def excess_green_mask(rgb: np.ndarray) -> np.ndarray:
    """
    rgb: HxWx3 uint8 or float array (0-255 or 0-1).
    Excess Green Index (ExG = 2G - R - B) is a classic, fast way to separate
    green vegetation from soil/background in RGB-only drone imagery (no NIR needed).
    """
    arr = rgb.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    exg = 2 * g - r - b
    # normalize 0-255 and Otsu-threshold to get a binary vegetation mask
    exg_norm = cv2.normalize(exg, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(exg_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def count_plants(rgb: np.ndarray, min_plant_area_px: int = 15,
                  min_distance_px: int = 8) -> dict:
    """
    Detects individual plant blobs via vegetation segmentation + distance-transform
    watershed (splits touching canopies at their narrowest connection point).

    Returns: {count, mean_canopy_area_px, centroids_px: [(row, col), ...], labeled_mask}
    """
    veg_mask = excess_green_mask(rgb)

    # clean small noise speckles
    kernel = np.ones((3, 3), np.uint8)
    veg_mask = cv2.morphologyEx(veg_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # distance transform: pixels far from background = plant centers
    distance = ndi.distance_transform_edt(veg_mask)
    coords = peak_local_max(distance, min_distance=min_distance_px, labels=veg_mask)
    peak_mask = np.zeros(distance.shape, dtype=bool)
    peak_mask[tuple(coords.T)] = True
    markers, _ = ndi.label(peak_mask)

    labels = watershed(-distance, markers, mask=veg_mask)

    plant_ids = [l for l in np.unique(labels) if l != 0]
    centroids = []
    areas = []
    for pid in plant_ids:
        blob = labels == pid
        area = int(blob.sum())
        if area < min_plant_area_px:
            continue
        ys, xs = np.where(blob)
        centroids.append((float(ys.mean()), float(xs.mean())))
        areas.append(area)

    return {
        "count": len(centroids),
        "mean_canopy_area_px": float(np.mean(areas)) if areas else 0.0,
        "centroids_px": centroids,
        "areas_px": areas,
    }


def pixel_centroids_to_geojson_points(centroids_px: list, areas_px: list, transform, crs=None) -> list[dict]:
    """Converts detected plant centroid pixel coords to georeferenced GeoJSON points
    with estimated canopy area (m^2), using the raster's affine transform.

    If crs is given and isn't already EPSG:4326, points are reprojected into
    EPSG:4326 so they line up with boundaries, plots, and the map — same fix
    applied to plots.py's extract_plots_from_drone_mask()."""
    # pixel size in map units (assumes transform is in meters, e.g. a UTM-projected
    # drone orthomosaic — the normal case for stitched drone output)
    px_area_m2 = abs(transform.a * transform.e)

    points = []
    for (row, col), area_px in zip(centroids_px, areas_px):
        gx, gy = xy(transform, row, col)
        geom = mapping(Point(gx, gy))
        points.append({
            "geometry": geom,
            "canopy_area_m2": round(area_px * px_area_m2, 4),
        })

    if crs is not None and str(crs) != "EPSG:4326":
        from rasterio.warp import transform_geom
        for p in points:
            p["geometry"] = transform_geom(crs, "EPSG:4326", p["geometry"])

    return points


def aggregate_plants_per_plot(plant_points: list[dict], plots: list[dict]) -> list[dict]:
    """
    Spatially joins detected plant points (EPSG:4326, from pixel_centroids_to_geojson_points)
    into plot polygons (EPSG:4326, from the plots table / plot extraction), so plant counts
    can be reported per-plot instead of only field-wide — matching Solvi's 'statistics for
    plant counts' as part of per-plot Zonal Statistics.

    plant_points: list of {geometry: GeoJSON Point, canopy_area_m2}
    plots: list of {plot_id, row, col, geometry: GeoJSON Polygon, ...}
    Returns one entry per plot: {plot_id, row, col, plant_count, mean_canopy_area_m2}
    """
    plot_polys = [(p["plot_id"], p["row"], p["col"], shape(p["geometry"])) for p in plots]
    point_geoms = [(shape(pt["geometry"]), pt["canopy_area_m2"]) for pt in plant_points]

    results = []
    for plot_id, row, col, poly in plot_polys:
        matched_areas = [area for pt, area in point_geoms if poly.contains(pt)]
        results.append({
            "plot_id": plot_id,
            "row": row,
            "col": col,
            "plant_count": len(matched_areas),
            "mean_canopy_area_m2": round(float(np.mean(matched_areas)), 4) if matched_areas else None,
        })
    return results
