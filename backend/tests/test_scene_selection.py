"""Scene search: only the best N scenes that cover the plot, clearest first.
The catalog client is faked, so no network is needed."""
import sys
import types

import pytest

import stac_client

CLOUD_ORDER = lambda scenes: [s["cloud_cover"] for s in scenes]


def cand(i, cloud, coverage=1.0, date=None):
    return {"id": f"S{i}", "cloud_cover": cloud, "coverage": coverage,
            "date": date or f"2026-09-{(i % 28) + 1:02d}T10:00:00Z"}


def test_selection_caps_filters_partials_and_orders_by_cloud():
    cands = [cand(i, cloud=i % 30, coverage=1.0 if i % 3 else 0.5) for i in range(40)]
    chosen, n_cover = stac_client.select_best_scenes(cands, limit=10)
    assert len(chosen) == 10
    assert all(c["coverage"] >= stac_client.MIN_COVERAGE for c in chosen)          # partials never shown
    assert CLOUD_ORDER(chosen) == sorted(CLOUD_ORDER(chosen))                       # clearest first
    assert n_cover == sum(1 for c in cands if c["coverage"] >= stac_client.MIN_COVERAGE)


def test_ties_prefer_the_newer_scene_and_unknown_cloud_goes_last():
    cands = [cand(1, 5, date="2026-08-01T00:00:00Z"), cand(2, 5, date="2026-09-20T00:00:00Z"),
             cand(3, None), cand(4, 1)]
    chosen, _ = stac_client.select_best_scenes(cands, limit=10)
    assert [c["id"] for c in chosen] == ["S4", "S2", "S1", "S3"]


def test_fewer_scenes_than_the_cap_returns_them_all():
    chosen, n = stac_client.select_best_scenes([cand(1, 3), cand(2, 8)], limit=10)
    assert len(chosen) == 2 and n == 2


def test_no_fully_covering_scene_returns_empty_but_reports_it():
    chosen, n = stac_client.select_best_scenes([cand(1, 3, coverage=0.4)], limit=10)
    assert chosen == [] and n == 0


# ---------------------------------------------------------------------------
# End to end through search_scenes_detailed with a fake STAC client
# ---------------------------------------------------------------------------

class FakeAsset:
    def __init__(self, href):
        self.href = href


class FakeItem:
    def __init__(self, i):
        self.id = f"S2_{i}"
        self.index = i
        self.geometry = {"index": i}
        self.properties = {"datetime": f"2026-09-{(i % 28) + 1:02d}T10:00:00Z",
                           "eo:cloud_cover": float(i % 30), "platform": "sentinel-2a"}
        self.assets = {k: FakeAsset(f"https://x/{i}/{k}") for k in ("B02", "B03", "B04", "B05", "B08", "SCL")}


def install_fakes(monkeypatch, n_items=40):
    seen = {"search_kwargs": None, "signed": 0}
    items = [FakeItem(i) for i in range(n_items)]

    class Search:
        def items(self):
            return iter(items)

    class Client:
        @staticmethod
        def open(url):
            return Client()

        def search(self, **kw):
            seen["search_kwargs"] = kw
            return Search()

    pc = types.ModuleType("planetary_computer")

    def sign(item):
        seen["signed"] += 1
        return item
    pc.sign = sign
    pst = types.ModuleType("pystac_client")
    pst.Client = Client
    monkeypatch.setitem(sys.modules, "planetary_computer", pc)
    monkeypatch.setitem(sys.modules, "pystac_client", pst)
    # footprint coverage: every third scene only half covers the plot
    monkeypatch.setattr(stac_client, "coverage_fraction",
                        lambda geom, boundary: 0.5 if geom["index"] % 3 == 0 else 1.0)
    monkeypatch.setattr(stac_client, "_reflectance_offset", lambda item: -0.1)
    return seen, items


def test_search_returns_best_ten_covering_scenes_not_everything(monkeypatch):
    seen, items = install_fakes(monkeypatch, n_items=40)
    out = stac_client.search_scenes_detailed({"type": "Polygon"}, "2026-08-01", "2026-09-30", 30.0)
    assert len(out["scenes"]) == 10
    assert out["checked"] == 40
    assert out["fully_covering"] == sum(1 for i in range(40) if i % 3 != 0)
    assert all(s["coverage_pct"] >= 99 for s in out["scenes"])
    clouds = [s["cloud_cover"] for s in out["scenes"]]
    assert clouds == sorted(clouds)
    assert seen["signed"] == 10                                      # only the winners are signed
    kw = seen["search_kwargs"]
    assert kw["max_items"] == stac_client.CANDIDATE_POOL             # the real cap: limit= is just page size
    assert kw["query"] == {"eo:cloud_cover": {"lt": 30.0}}


def test_custom_cap_and_wrapper(monkeypatch):
    install_fakes(monkeypatch)
    assert len(stac_client.search_scenes({"type": "Polygon"}, "a", "b", 30.0, limit=4)) == 4


def test_coverage_fraction_with_real_geometry():
    pytest.importorskip("shapely")
    farm = {"type": "Polygon", "coordinates": [[[7.40, 9.07], [7.41, 9.07], [7.41, 9.08], [7.40, 9.08], [7.40, 9.07]]]}
    full = {"type": "Polygon", "coordinates": [[[7.0, 9.0], [8.0, 9.0], [8.0, 10.0], [7.0, 10.0], [7.0, 9.0]]]}
    half = {"type": "Polygon", "coordinates": [[[7.405, 9.0], [8.0, 9.0], [8.0, 10.0], [7.405, 10.0], [7.405, 9.0]]]}
    miss = {"type": "Polygon", "coordinates": [[[8.0, 9.0], [8.5, 9.0], [8.5, 10.0], [8.0, 10.0], [8.0, 9.0]]]}
    assert stac_client.coverage_fraction(full, farm) == pytest.approx(1.0)
    assert stac_client.coverage_fraction(half, farm) == pytest.approx(0.5)
    assert stac_client.coverage_fraction(miss, farm) == 0.0
