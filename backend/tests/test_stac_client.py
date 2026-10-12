"""Cloud masking, reflectance offset and mixed-resolution alignment, using local GeoTIFFs."""
import numpy as np
import pytest
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin
from rasterio.warp import transform_geom
from shapely.geometry import box, mapping

import stac_client

CRS = "EPSG:32632"


@pytest.fixture()
def scene(tmp_path):
    ox, oy = Transformer.from_crs("EPSG:4326", CRS, always_xy=True).transform(7.39, 9.08)
    ox, oy = round(ox / 20) * 20, round(oy / 20) * 20

    def write(name, arr, res):
        path = tmp_path / f"{name}.tif"
        with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                           dtype="uint16", crs=CRS, transform=from_origin(ox, oy, res, res)) as d:
            d.write(arr.astype("uint16"), 1)
        return str(path)

    n10, n20 = 400, 200
    hrefs = {
        "red": write("red", np.full((n10, n10), (0.05 + 0.1) * 10000), 10),    # baseline>=04.00 DN offset
        "nir": write("nir", np.full((n10, n10), (0.45 + 0.1) * 10000), 10),
        "rededge": write("re", (0.10 + np.tile(np.arange(n20), (n20, 1)) * 0.001 + 0.1) * 10000, 20),
    }
    scl = np.full((n20, n20), 4)
    scl[40:60, 40:60] = 9       # cloud   : 20x20 @20 m = 1600 px @10 m
    scl[60:70, 60:70] = 3       # shadow  : 10x10 @20 m =  400 px @10 m
    hrefs["scl"] = write("scl", scl, 20)
    field = transform_geom(CRS, "EPSG:4326", mapping(box(ox + 500, oy - 1500, ox + 1500, oy - 500)))
    return hrefs, field


def test_offset_is_applied(scene):
    hrefs, field = scene
    bands, _ = stac_client.fetch_and_clip_bands(hrefs, field, ["red", "nir"], reflectance_offset=-0.1)
    assert np.nanmean(bands["red"]) == pytest.approx(0.05, abs=1e-3)
    assert np.nanmean(bands["nir"]) == pytest.approx(0.45, abs=1e-3)


def test_cloud_and_shadow_are_masked(scene):
    hrefs, field = scene
    _, meta = stac_client.fetch_and_clip_bands(hrefs, field, ["red", "nir"], reflectance_offset=-0.1)
    assert meta["field_pixels"] - meta["valid_pixels"] == 2000
    assert meta["valid_fraction"] == pytest.approx(0.8)
    _, raw = stac_client.fetch_and_clip_bands(hrefs, field, ["red", "nir"], reflectance_offset=-0.1,
                                               apply_cloud_mask=False)
    assert raw["valid_fraction"] == 1.0


def test_mixed_resolution_bands_align(scene):
    hrefs, field = scene
    bands, _ = stac_client.fetch_and_clip_bands(hrefs, field, ["rededge", "nir"], reflectance_offset=-0.1)
    assert bands["rededge"].shape == bands["nir"].shape
    left = np.nanmean(bands["rededge"][:, :4])
    assert left == pytest.approx(0.125, abs=0.005)          # boundary starts 500 m = 25 cols in
    assert np.nanmean(bands["rededge"][:, -4:]) > left


def test_missing_band_is_a_clear_error(scene):
    hrefs, field = scene
    with pytest.raises(ValueError, match="missing required band"):
        stac_client.fetch_and_clip_bands(hrefs, field, ["swir1", "nir"])


def test_scl_valid_classes():
    scl = np.array([0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    assert stac_client.scl_valid_mask(scl).tolist() == [False, False, False, True, True, True, True,
                                                         False, False, False, False]
