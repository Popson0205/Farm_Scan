"""Areas on the ellipsoid (WGS 84), so a farm's hectares are right anywhere, not approximated from degrees."""
from pyproj import Geod
from shapely.geometry import shape

_GEOD = Geod(ellps="WGS84")


def area_m2(geometry) -> float:
    """Area of a GeoJSON geometry (or shapely geometry) in EPSG:4326, in square metres."""
    g = shape(geometry) if isinstance(geometry, dict) else geometry
    return abs(_GEOD.geometry_area_perimeter(g)[0])
