"""
Automatic plot extraction using a pretrained segmentation model (MobileSAM,
a lightweight distillation of Meta's Segment Anything Model) instead of the
classical excess-green vegetation mask in plots.py.

WHY THIS EXISTS
----------------
The vegetation-mask + contour-detection approach in plots.py assumes plots
are distinguished from their surroundings by canopy greenness. That holds for
young, well-spaced, uniformly-planted trial plots on bare soil (see the
synthetic sample data), but breaks down for real plantation imagery where
plots are instead delineated by dirt access roads/tracks cutting through
otherwise-continuous vegetation or bare-soil blocks — greenness alone can't
separate "plot A" from "plot B" in that case; a road detector or a general
learned notion of "distinct region" is needed instead.

MobileSAM (https://github.com/ChaoningZhang/MobileSAM) is a promptable
segmentation model distilled from Meta's SAM, trained on the 11M-image SA-1B
dataset. It has never seen farm imagery specifically, but its "automatic mask
generation" mode segments an image into all its visually distinct regions —
which, on plantation/field imagery, tends to align with individual
plot/parcel boundaries far better than a hand-crafted color threshold,
because it's responding to texture, tone, and edge structure jointly rather
than one fixed rule. This is a genuine zero-shot use of a pretrained model,
not a farm-specific trained model — see the module docstring notes below on
what fine-tuning on a real field-boundary dataset (e.g. Fields of the World,
AI4Boundaries, Delineate Anything's FBIS-22M) would add on top of this.

REQUIREMENTS
------------
This method needs `torch` and `timm` installed, plus the MobileSAM weights
present at backend/models/mobile_sam.pt. The weights are NOT committed to
git (39MB is over GitHub's web-upload UI limit, and binaries bloat git
history) -- run `backend/scripts/download_sam_weights.py` (or the .sh
equivalent) once after cloning to fetch them from the upstream MobileSAM
repo. All of this is optional -- the app runs fine without it, and
/api/plots/extract-from-drone simply reports this method as unavailable via
GET /api/plots/extraction-methods if the weights or dependencies are
missing, falling back to the vegetation-mask method.

PERFORMANCE NOTE
-----------------
CPU inference is slow (tens of seconds per image at modest settings). For
production use with real orthomosaics, run this on a GPU instance, or reduce
`points_per_side` / restrict to a downsampled preview resolution and only
run full-resolution SAM on the final confirmed selection.
"""
import sys
from pathlib import Path

import numpy as np
import cv2
from shapely.geometry import shape, mapping, Polygon
from rasterio.warp import transform_geom

_VENDOR_DIR = Path(__file__).parent / "vendor"
_WEIGHTS_PATH = Path(__file__).parent / "models" / "mobile_sam.pt"

_sam_mask_generator = None  # lazy-loaded singleton


def sam_available() -> tuple[bool, str]:
    """Checks whether the SAM extraction path can run at all, without
    actually loading the (somewhat slow-to-load) model."""
    if not _WEIGHTS_PATH.exists():
        return False, (
            f"Model weights not found at {_WEIGHTS_PATH}. Run: "
            f"python3 backend/scripts/download_sam_weights.py "
            f"(or bash backend/scripts/download_sam_weights.sh) to fetch them (~39MB, "
            f"not committed to git to keep the repo small)."
        )
    try:
        import torch  # noqa: F401
        import timm  # noqa: F401
    except ImportError as e:
        return False, f"Missing dependency: {e}. Install with: pip install torch timm"
    return True, ""


def _load_mask_generator(points_per_side: int = 16):
    global _sam_mask_generator
    if _sam_mask_generator is not None:
        return _sam_mask_generator

    if str(_VENDOR_DIR) not in sys.path:
        sys.path.insert(0, str(_VENDOR_DIR))

    import torch
    from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_t"](checkpoint=str(_WEIGHTS_PATH))
    sam.to(device=device)
    sam.eval()

    _sam_mask_generator = SamAutomaticMaskGenerator(
        sam,
        points_per_side=points_per_side,
        pred_iou_thresh=0.85,
        stability_score_thresh=0.88,
        min_mask_region_area=800,
    )
    return _sam_mask_generator


def _mask_to_polygon(mask: np.ndarray) -> Polygon | None:
    """Converts a boolean segmentation mask to a single (largest-contour)
    shapely polygon in pixel coordinates, simplified to reduce vertex noise
    from the mask's raster edges."""
    mask_u8 = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if len(largest) < 3:
        return None
    coords = largest.reshape(-1, 2)
    poly = Polygon(coords)
    if not poly.is_valid or poly.area == 0:
        poly = poly.buffer(0)
    if poly.is_empty:
        return None
    # simplify to cut down on jagged mask-edge vertices while preserving shape
    tolerance = max(1.0, (poly.area ** 0.5) * 0.01)
    return poly.simplify(tolerance, preserve_topology=True)


def extract_plots_via_sam(
    rgb: np.ndarray,
    transform,
    crs,
    min_area_frac: float = 0.004,
    max_area_frac: float = 0.30,
    border_frac_thresh: float = 0.5,
    points_per_side: int = 16,
    max_dim: int = 768,
) -> list[dict]:
    """
    Segments an orthomosaic (or oblique photo, for quick experimentation —
    see the caveat about oblique imagery in extract-from-drone's docstring)
    into individual plots using MobileSAM's automatic mask generation, then
    filters out masks that are clearly not individual plots:
      - too small a fraction of the image (noise / sub-object fragments)
      - too large a fraction of the image (the whole field/background, not
        one plot)
      - touching the image border heavily (partial/cut-off objects, sky,
        horizon haze in oblique shots)

    rgb: HxWx3 uint8 array (drone RGB)
    transform: rasterio Affine transform for the source raster, in ORIGINAL
               (full) pixel space — this function downsamples internally for
               CPU memory/speed and scales results back up before applying it
    crs: the raster's native CRS, for reprojecting results to EPSG:4326
    max_dim: longest side to downsample to before running SAM. CPU inference
             on the full resolution of a real orthomosaic (which can be
             thousands of pixels per side) is both extremely slow and prone
             to out-of-memory kills on modest hardware; this keeps memory use
             bounded. Raise this (or run on a GPU instance) for finer detail.
    Returns the same shape as plots.extract_plots_from_drone_mask(): a list of
    {plot_id, row, col, geometry (GeoJSON, EPSG:4326), area_m2, confidence}.
    """
    ok, reason = sam_available()
    if not ok:
        raise RuntimeError(f"SAM extraction unavailable: {reason}")

    orig_h, orig_w = rgb.shape[:2]
    scale = min(1.0, max_dim / max(orig_h, orig_w))
    if scale < 1.0:
        small_w, small_h = int(orig_w * scale), int(orig_h * scale)
        rgb_small = cv2.resize(rgb, (small_w, small_h), interpolation=cv2.INTER_AREA)
    else:
        rgb_small = rgb

    mask_generator = _load_mask_generator(points_per_side=points_per_side)
    h, w = rgb_small.shape[:2]

    masks = mask_generator.generate(rgb_small)

    # scale factor to map downsampled-pixel coords back to the original
    # raster's pixel space, so `transform` (defined for the full-res image)
    # applies correctly
    inv_scale = 1.0 / scale if scale < 1.0 else 1.0

    candidates = []
    for m in masks:
        seg = m["segmentation"]
        area_frac = seg.sum() / (h * w)
        if area_frac < min_area_frac or area_frac > max_area_frac:
            continue
        border = np.concatenate([seg[0, :], seg[-1, :], seg[:, 0], seg[:, -1]])
        if border.mean() > border_frac_thresh:
            continue

        poly_px = _mask_to_polygon(seg)
        if poly_px is None or poly_px.area < 4:
            continue

        # pixel (col,row) in the DOWNSAMPLED image -> original full-res pixel
        # space -> map coords using the affine transform, then to EPSG:4326
        def px_to_map(coords):
            return [transform * (x * inv_scale, y * inv_scale) for x, y in coords]

        try:
            map_coords = px_to_map(list(poly_px.exterior.coords))
            poly_map = Polygon(map_coords)
        except Exception:
            continue
        if not poly_map.is_valid or poly_map.is_empty:
            continue

        area_m2 = float(poly_map.area)  # valid if raster CRS is projected (meters)

        geom_4326 = transform_geom(crs, "EPSG:4326", mapping(poly_map))
        poly_4326 = shape(geom_4326)

        candidates.append({
            "geometry": mapping(poly_4326),
            "area_px": int(seg.sum() * inv_scale * inv_scale),
            "area_m2": round(area_m2, 2),
            "confidence": round(float(m.get("stability_score", 0.0)), 3),
            "_centroid": (poly_4326.centroid.x, poly_4326.centroid.y),
        })

    candidates.sort(key=lambda c: c["area_px"], reverse=True)

    # reuse plots.py's row/col clustering so results plug into the same
    # plots table / stats / exports as the vegetation-mask method
    from plots import _assign_grid_positions
    return _assign_grid_positions(candidates)
