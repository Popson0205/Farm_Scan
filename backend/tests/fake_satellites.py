"""A stand-in for Microsoft Planetary Computer: real Cloud-Optimized GeoTIFFs written to a temporary folder, served
through a fake STAC catalogue, so the whole satellite pipeline (search, coverage, band reads and warping, cloud mask,
reflectance offset, indices, radar) runs on real raster files without the network.

Sentinel-2 tiles are in UTM 32N like the real ones over central Nigeria (10 m bands, 20 m red edge / SWIR / SCL), with
a +1000 DN offset (processing baseline 05.x). Each date's crop follows a season curve; a weak patch sits in the field's
south-west; some dates carry a cloud over part of the field. Sentinel-1 RTC tiles carry VV/VH in linear gamma0."""
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform as warp_transform

CRS = "EPSG:32632"


def season_ndvi(d: date) -> float:
    t = (d - date(2026, 6, 1)).days
    up = 1 / (1 + np.exp(-(t - 35) / 7))
    down = 1 / (1 + np.exp((d - date(2026, 10, 15)).days / 5))
    return 0.2 + 0.55 * up * down


class Asset:
    def __init__(self, href, extra=None):
        self.href, self.extra_fields = href, extra or {}


class Item:
    def __init__(self, id, geometry, props, assets):
        self.id, self.geometry, self.properties, self.assets = id, geometry, props, assets


def _write(path: Path, arr, transform, dtype, nodata=0):
    with rasterio.open(path, "w", driver="GTiff", width=arr.shape[1], height=arr.shape[0], count=1, dtype=dtype,
                       crs=CRS, transform=transform, nodata=nodata, tiled=True, blockxsize=256, blockysize=256) as ds:
        ds.write(arr.astype(dtype), 1)


class FakeCatalog:
    """Tiles around a farm's centre: write once per date on demand, keep the items."""

    def __init__(self, root: Path, lon: float, lat: float, size_m: int = 5120):
        self.root = root
        (x,), (y,) = warp_transform("EPSG:4326", CRS, [lon], [lat])
        self.x0, self.y0 = round(x - size_m / 2, -1), round(y + size_m / 2, -1)
        self.size = size_m
        self.s2: dict[str, Item] = {}
        self.s1: dict[str, Item] = {}
        self.cloudy: set[str] = set()
        self.sign_calls = 0

    def footprint(self) -> dict:
        xs = [self.x0, self.x0 + self.size, self.x0 + self.size, self.x0, self.x0]
        ys = [self.y0, self.y0, self.y0 - self.size, self.y0 - self.size, self.y0]
        lons, lats = warp_transform(CRS, "EPSG:4326", xs, ys)
        return {"type": "Polygon", "coordinates": [list(zip(lons, lats))]}

    def add_s2(self, d: date, cloud_cover: float = 5.0, cloud_over_field: bool = False, weak_patch: bool = True) -> Item:
        sid = f"S2B_MSIL2A_{d:%Y%m%d}T095029_R079_T32PNR"
        folder = self.root / sid
        folder.mkdir(parents=True, exist_ok=True)
        n10 = self.size // 10
        rng = np.random.default_rng(d.toordinal())
        v = season_ndvi(d)
        ndvi = np.full((n10, n10), v) + rng.normal(0, 0.01, (n10, n10))
        if weak_patch:                                      # a weak patch south-west of the centre
            c = n10 // 2
            ndvi[c + 5:c + 20, c - 20:c - 5] -= 0.25
        ndvi = np.clip(ndvi, -0.2, 0.95)
        red = np.full_like(ndvi, 0.06)
        nir = red * (1 + ndvi) / (1 - ndvi)
        green, blue = red * 1.2, red * 0.8
        dn = lambda r: np.clip(r * 10000 + 1000, 1, 20000)          # noqa: E731  (baseline 05.x: +1000 offset)
        t10 = from_origin(self.x0, self.y0, 10, 10)
        t20 = from_origin(self.x0, self.y0, 20, 20)
        for key, arr in (("B02", blue), ("B03", green), ("B04", red), ("B08", nir)):
            _write(folder / f"{key}.tif", dn(arr), t10, "uint16")
        down = lambda a: a.reshape(n10 // 2, 2, n10 // 2, 2).mean(axis=(1, 3))   # noqa: E731
        _write(folder / "B05.tif", dn(down(red * 1.5 + nir * 0.2)), t20, "uint16")
        _write(folder / "B11.tif", dn(down(np.full_like(red, 0.18) - 0.1 * ndvi)), t20, "uint16")
        scl = np.full((n10 // 2, n10 // 2), 4, dtype="uint8")
        if cloud_over_field:                               # cloud over the north half of the field
            c = n10 // 4
            scl[c - 30:c, c - 30:c + 30] = 9
            self.cloudy.add(d.isoformat())
        _write(folder / "SCL.tif", scl, t20, "uint8")
        assets = {k: Asset(str(folder / f"{k}.tif"), {"raster:bands": [{"offset": -0.1, "scale": 0.0001}]} if k in ("B04", "B08") else {})
                  for k in ("B02", "B03", "B04", "B05", "B08", "B11", "SCL")}
        item = Item(sid, self.footprint(), {"datetime": f"{d.isoformat()}T09:50:29Z", "eo:cloud_cover": cloud_cover,
                                             "platform": "Sentinel-2B", "s2:processing_baseline": "05.11"}, assets)
        self.s2[d.isoformat()] = item
        return item

    def add_s1(self, d: date) -> Item:
        sid = f"S1A_IW_GRDH_1SDV_{d:%Y%m%d}T052000_rtc"
        folder = self.root / sid
        folder.mkdir(parents=True, exist_ok=True)
        n10 = self.size // 10
        rng = np.random.default_rng(d.toordinal() + 7)
        v = season_ndvi(d)
        vv = np.clip(0.12 * rng.gamma(4.0, 0.25, (n10, n10)), 1e-4, None)         # speckle
        vh = np.clip(vv * (0.06 + 0.25 * (v - 0.2) / 0.55) * rng.gamma(4.0, 0.25, (n10, n10)), 1e-5, None)
        t = from_origin(self.x0, self.y0, 10, 10)
        _write(folder / "vv.tif", vv, t, "float32", nodata=-32768)
        _write(folder / "vh.tif", vh, t, "float32", nodata=-32768)
        item = Item(sid, self.footprint(), {"datetime": f"{d.isoformat()}T05:20:00Z", "sat:orbit_state": "ascending",
                                             "platform": "SENTINEL-1A"},
                    {"vv": Asset(str(folder / "vv.tif")), "vh": Asset(str(folder / "vh.tif"))})
        self.s1[d.isoformat()] = item
        return item

    # the parts of pystac_client / planetary_computer the code uses
    def search(self, collections, datetime, query=None, **kw):
        lo, hi = datetime.split("/")
        coll = collections[0]
        pool = self.s1 if coll == "sentinel-1-rtc" else self.s2
        items = [it for d, it in sorted(pool.items()) if lo[:10] <= d <= hi[:10]]
        if query and "eo:cloud_cover" in query:
            items = [it for it in items if it.properties["eo:cloud_cover"] < query["eo:cloud_cover"]["lt"]]

        class Search:
            def items(self_inner):
                return iter(items)
        return Search()

    def install(self, monkeypatch):
        import planetary_computer
        import pystac_client

        cat = self

        class Client:
            @staticmethod
            def open(url):
                return cat

        def sign(x):
            cat.sign_calls += 1
            return x
        monkeypatch.setattr(pystac_client, "Client", Client)
        monkeypatch.setattr(planetary_computer, "sign", sign)
        return self
