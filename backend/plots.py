"""
Plot layout: splits a farm/field boundary into a regular grid of trial plots
(the common case for row-crop / field-trial layouts). Plots can then be edited,
added, deleted, or replaced by an imported SHP/KML layer (see plot_import.py).
"""
import numpy as np
from shapely.geometry import box, shape, mapping


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
