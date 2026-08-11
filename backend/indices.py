"""
Vegetation & water index calculations from Sentinel-2 surface reflectance bands.

Sentinel-2 band reference (10m/20m bands used here):
  B02 = Blue     B03 = Green     B04 = Red
  B05 = Red Edge 1   B08 = NIR    B11 = SWIR 1

All formulas expect float32 arrays scaled 0-1 (reflectance) or raw DN/10000.
"""
import numpy as np

EPS = 1e-8


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Divide with protection against zero-division and NaN propagation."""
    denom = np.where(np.abs(denominator) < EPS, EPS, denominator)
    result = numerator / denom
    return np.clip(result, -1.0, 1.0)


def ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Normalized Difference Vegetation Index. Range -1..1. Higher = healthier/denser vegetation."""
    return _safe_ratio(nir - red, nir + red)


def ndre(red_edge: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Normalized Difference Red Edge. Less prone to saturation than NDVI in dense canopy;
    better for detecting nitrogen/chlorophyll stress in mature crops."""
    return _safe_ratio(nir - red_edge, nir + red_edge)


def ndwi(green: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """Normalized Difference Water Index (McFeeters). Positive values = open water / high moisture surfaces."""
    return _safe_ratio(green - nir, green + nir)


def ndmi(nir: np.ndarray, swir1: np.ndarray) -> np.ndarray:
    """Normalized Difference Moisture Index. Sensitive to vegetation/canopy water content (crop water stress)."""
    return _safe_ratio(nir - swir1, nir + swir1)


def savi(red: np.ndarray, nir: np.ndarray, L: float = 0.5) -> np.ndarray:
    """Soil-Adjusted Vegetation Index. L=0.5 works well for moderate vegetation cover;
    reduces soil brightness influence vs plain NDVI on sparse/young crops."""
    denom = nir + red + L
    denom = np.where(np.abs(denom) < EPS, EPS, denom)
    result = ((nir - red) / denom) * (1 + L)
    return np.clip(result, -1.0, 1.0)


def evi(blue: np.ndarray, red: np.ndarray, nir: np.ndarray,
        G: float = 2.5, C1: float = 6.0, C2: float = 7.5, L: float = 1.0) -> np.ndarray:
    """Enhanced Vegetation Index. Corrects for atmospheric haze and canopy background,
    performs better than NDVI in high-biomass areas."""
    denom = nir + C1 * red - C2 * blue + L
    denom = np.where(np.abs(denom) < EPS, EPS, denom)
    result = G * (nir - red) / denom
    return np.clip(result, -1.0, 1.0)


INDEX_REGISTRY = {
    "ndvi": {"fn": ndvi, "bands": ["red", "nir"], "label": "NDVI (Vegetation)",
             "colormap": "RdYlGn", "range": (-0.2, 0.9)},
    "ndre": {"fn": ndre, "bands": ["rededge", "nir"], "label": "NDRE (Crop Vigor / Nitrogen)",
              "colormap": "RdYlGn", "range": (-0.1, 0.6)},
    "ndwi": {"fn": ndwi, "bands": ["green", "nir"], "label": "NDWI (Surface Water)",
             "colormap": "Blues", "range": (-0.3, 0.6)},
    "ndmi": {"fn": ndmi, "bands": ["nir", "swir1"], "label": "NDMI (Crop Water Stress)",
             "colormap": "BrBG", "range": (-0.4, 0.5)},
    "savi": {"fn": savi, "bands": ["red", "nir"], "label": "SAVI (Soil-Adjusted Vegetation)",
             "colormap": "RdYlGn", "range": (-0.2, 0.7)},
    "evi": {"fn": evi, "bands": ["blue", "red", "nir"], "label": "EVI (Enhanced Vegetation)",
            "colormap": "RdYlGn", "range": (-0.2, 0.8)},
}


def compute_index(name: str, band_arrays: dict) -> np.ndarray:
    """band_arrays: dict mapping band role name (e.g. 'red', 'nir') -> np.ndarray"""
    if name not in INDEX_REGISTRY:
        raise ValueError(f"Unknown index '{name}'. Available: {list(INDEX_REGISTRY)}")
    spec = INDEX_REGISTRY[name]
    args = [band_arrays[b] for b in spec["bands"]]
    return spec["fn"](*args)


def zonal_stats(index_array: np.ndarray, mask: np.ndarray) -> dict:
    """Compute summary stats for an index array restricted to a boolean field mask."""
    valid = index_array[mask & ~np.isnan(index_array)]
    if valid.size == 0:
        return {"mean": None, "min": None, "max": None, "std": None, "pixel_count": 0}
    return {
        "mean": float(np.mean(valid)),
        "min": float(np.min(valid)),
        "max": float(np.max(valid)),
        "std": float(np.std(valid)),
        "pixel_count": int(valid.size),
    }
