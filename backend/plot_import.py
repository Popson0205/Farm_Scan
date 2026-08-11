"""
Imports existing plot boundaries from SHP (zipped shapefile) or KML files,
for trials where plots were already digitized elsewhere (QGIS, a trial
management system, etc.) — Solvi's "Import existing plots from SHP and KML
files" as part of Zonal Statistics / Plot Extraction.
"""
import io
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import geopandas as gpd
from shapely.geometry import mapping


def _row_col_from_attrs_or_grid(gdf: gpd.GeoDataFrame) -> list[tuple]:
    """
    If the imported file already has row/col-like attribute columns, use them;
    otherwise fall back to spatially clustering centroids into a grid, the same
    approach used for auto-detected plots (plots.py's _assign_grid_positions),
    so imported plots still get sensible labels for display/export ordering.
    """
    cols_lower = {c.lower(): c for c in gdf.columns}
    row_col = None
    col_col = None
    for candidate in ("row", "row_idx", "row_id", "plot_row"):
        if candidate in cols_lower:
            row_col = cols_lower[candidate]
            break
    for candidate in ("col", "col_idx", "col_id", "plot_col", "column"):
        if candidate in cols_lower:
            col_col = cols_lower[candidate]
            break

    if row_col and col_col:
        return [(int(r[row_col]), int(r[col_col])) for _, r in gdf.iterrows()]
    return None


def parse_plots_file(content: bytes, suffix: str) -> list[dict]:
    """
    content: raw uploaded file bytes
    suffix: '.zip' (zipped shapefile), '.shp' (bare shapefile — unusual but
            handled), or '.kml'
    Returns a list of {plot_id, row, col, geometry (GeoJSON, EPSG:4326), area_m2}
    ready to insert into the plots table.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        if suffix == ".zip":
            zf = zipfile.ZipFile(io.BytesIO(content))
            zf.extractall(tmpdir)
            shp_candidates = list(tmpdir.rglob("*.shp"))
            if not shp_candidates:
                raise ValueError("No .shp file found inside the uploaded zip")
            read_path = shp_candidates[0]
        elif suffix == ".kml":
            # fiona's KML driver is read-only-disabled by default in some builds;
            # explicitly enable it for this read.
            import fiona
            fiona.drvsupport.supported_drivers["KML"] = "rw"
            read_path = tmpdir / "import.kml"
            read_path.write_bytes(content)
        else:  # bare .shp — geopandas needs the sibling files too, best effort
            read_path = tmpdir / "import.shp"
            read_path.write_bytes(content)

        gdf = gpd.read_file(read_path)

    if gdf.empty:
        return []

    # keep only polygonal geometries (ignore stray points/lines in the same layer)
    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]
    if gdf.empty:
        return []

    if gdf.crs is None:
        # no CRS info in the file — assume it's already lat/lon, the common case
        # for hand-digitized KML and many trial-management SHP exports
        gdf = gdf.set_crs("EPSG:4326")
    elif str(gdf.crs) != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    row_cols = _row_col_from_attrs_or_grid(gdf)

    plot_list = []
    for i, (_, r) in enumerate(gdf.iterrows()):
        geom = r.geometry
        lat = geom.centroid.y
        m_per_deg_lat = 111320.0
        m_per_deg_lon = 111320.0 * np.cos(np.radians(lat))
        area_m2 = round(geom.area * m_per_deg_lat * m_per_deg_lon, 2)

        if row_cols:
            row, col = row_cols[i]
        else:
            row, col = 0, i

        plot_list.append({
            "plot_id": i,
            "row": row,
            "col": col,
            "geometry": mapping(geom),
            "area_m2": area_m2,
        })

    return plot_list
