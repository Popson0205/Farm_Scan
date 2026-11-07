"""
Soil from ISRIC SoilGrids 2.0 (250 m, global, free, no key): used when the HWSD files are not installed, so crop
suitability works on any deployment. SoilGrids is finer than HWSD (250 m against ~1 km) but has no drainage,
salinity or base-saturation layers; those factors are then left out of the scoring and the report says so.

REST: https://rest.isric.org/soilgrids/v2.0/properties/query  (fair use: about 5 requests a minute; answers are
cached for 180 days, as soil does not change).
"""
import time

import requests
from shapely.geometry import shape

URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"
PROPS = ("sand", "silt", "clay", "soc", "phh2o", "cec", "bdod", "nitrogen", "cfvo")
DEPTHS = ("0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm")
# SoilGrids property -> (our key, its d_factor if the answer omits it). Dividing the mapped value by d_factor gives
# the target unit (%, pH, cmol/kg, kg/dm3, g/kg, vol %); organic carbon's target is g/kg, which EXTRA turns into %.
UNITS = {
    "sand": ("sand_pct", 10),            # g/kg -> %
    "silt": ("silt_pct", 10),
    "clay": ("clay_pct", 10),
    "soc": ("organic_carbon_pct", 10),   # dg/kg -> g/kg (then % below)
    "phh2o": ("ph_water", 10),           # pH x10
    "cec": ("cec_cmol_kg", 10),          # mmol(c)/kg -> cmol/kg
    "bdod": ("bulk_density", 100),       # cg/cm3 -> g/cm3
    "nitrogen": ("total_n_g_kg", 100),   # cg/kg -> g/kg
    "cfvo": ("coarse_frag_pct", 10),     # cm3/dm3 -> vol %
}
EXTRA = {"soc": 10}                       # g/kg -> %
CACHE_S = 180 * 86400


class SoilGridsUnavailable(RuntimeError):
    pass


def _fetch(lat: float, lon: float) -> dict:
    import weather
    key = f"soilgrids:{round(lat, 4)}:{round(lon, 4)}"
    hit = weather._cache_get(key, CACHE_S)
    if hit:
        return hit["payload"]
    params = [("lat", round(lat, 5)), ("lon", round(lon, 5)), ("value", "mean")] + [("property", p) for p in PROPS] \
        + [("depth", d) for d in DEPTHS]
    last = None
    for attempt in range(3):
        try:
            r = requests.get(URL, params=params, timeout=30)
            if r.status_code == 429 and attempt < 2:
                time.sleep(15)
                continue
            r.raise_for_status()
            j = r.json()
            weather._cache_put(key, j)
            return j
        except (requests.RequestException, ValueError) as e:
            last = e
    raise SoilGridsUnavailable(f"SoilGrids could not be reached ({last}). Try again in a few minutes.")


def layers_from(j: dict) -> dict:
    """{depth label: {our key: value}} from a SoilGrids answer (values missing where SoilGrids has none)."""
    out: dict[str, dict] = {}
    for layer in (j.get("properties") or {}).get("layers", []):
        name = layer.get("name")
        if name not in UNITS:
            continue
        key, div = UNITS[name]
        d_factor = (layer.get("unit_measure") or {}).get("d_factor") or div
        for d in layer.get("depths", []):
            v = (d.get("values") or {}).get("mean")
            if v is None:
                continue
            out.setdefault(d["label"], {})[key] = v / d_factor / EXTRA.get(name, 1)
    return out


def _depth_avg(layers: dict, labels: tuple[str, ...]) -> dict:
    acc: dict[str, list] = {}
    for lab in labels:
        a, b = (int(x) for x in lab.replace("cm", "").split("-"))
        for k, v in layers.get(lab, {}).items():
            s = acc.setdefault(k, [0.0, 0.0])
            s[0] += (b - a) * v
            s[1] += b - a
    return {k: round(s / t, 2) for k, (s, t) in acc.items() if t}


def soil_profile(boundary_geojson: dict) -> dict:
    from soil import usda_texture
    c = shape(boundary_geojson).representative_point()
    layers = layers_from(_fetch(c.y, c.x))
    top = _depth_avg(layers, ("0-5cm", "5-15cm", "15-30cm"))
    if not top or "clay_pct" not in top:
        raise SoilGridsUnavailable("SoilGrids has no soil data at this farm (water, built-up land, or outside its map).")
    sub = _depth_avg(layers, ("30-60cm", "60-100cm"))
    for p in (top, sub):
        if all(k in p for k in ("sand_pct", "silt_pct", "clay_pct")):
            p["texture_class"] = usda_texture(p["sand_pct"], p["silt_pct"], p["clay_pct"])
    return {
        "source": "ISRIC SoilGrids 2.0",
        "resolution": "250 m",
        "pixels_sampled": 1,
        "mapping_units": [],
        "soil_groups": {},
        "drainage_distribution": {},
        "drainage_class_weights": {},
        "topsoil_0_30cm": top,
        "subsoil_30_100cm": sub,
        "by_layer": {lab: {"depth_cm": [int(x) for x in lab.replace("cm", "").split("-")], **v} for lab, v in layers.items()},
        "soil_test_needed": True,
        "notes": ["Soil from SoilGrids (250 m, predicted from soil surveys and satellite covariates) at the farm's centre: "
                  "it has no drainage or salinity layers, so those are not scored.",
                  "Use for crop screening and planning. Fertilizer and lime rates need a laboratory soil test."],
    }
