#!/usr/bin/env python3
"""
One-time setup for the soil layer: turn the FAO HWSD v2.0 attribute table into the
small indexed SQLite file FarmScan queries, and check the raster is in place.

1) Download from the FAO soils portal (HWSD v2.0 page, "Download" links):
     * HWSD2 raster   (HWSD2_RASTER.zip -> HWSD2.bil + HWSD2.hdr [+ .prj])
     * HWSD2 database (HWSD2.mdb)  -- or any CSV export of its HWSD2_LAYERS table
   https://www.fao.org/soils-portal/data-hub/soil-maps-and-databases/harmonized-world-soil-database-v20/en/
2) Put HWSD2.bil (+ its .hdr/.prj) in backend/data/hwsd/
3) Build the database:

     # from the .mdb  (needs mdbtools:  apt install mdbtools  |  brew install mdbtools)
     python scripts/build_hwsd_sqlite.py --mdb /path/to/HWSD2.mdb

     # or from a CSV you exported yourself (any OS, e.g. via Access / LibreOffice)
     python scripts/build_hwsd_sqlite.py --layers-csv /path/to/HWSD2_LAYERS.csv

The script prints which expected columns it found. Column names in the FAO release
are matched case-insensitively; if one of the important ones is reported MISSING,
open the table, check what it is called in your release, and adjust NUMERIC_PROPS
in soil.py (a one-line change).
"""
import argparse
import io
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import pandas as pd

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))
import soil  # noqa: E402

REQUIRED = ["HWSD2_SMU_ID", "SHARE", "LAYER", "SAND", "SILT", "CLAY", "PH_WATER", "ORG_CARBON", "CEC_SOIL"]
CATEGORICAL = ["DRAINAGE", "WRB4", "FAO90", "WRB2", "TEXTURE_USDA"]


def read_from_mdb(mdb: Path) -> pd.DataFrame:
    if not shutil.which("mdb-export"):
        sys.exit("mdb-export not found. Install mdbtools (apt install mdbtools / brew install mdbtools), "
                 "or export HWSD2_LAYERS to CSV and use --layers-csv.")
    out = subprocess.run(["mdb-export", str(mdb), "HWSD2_LAYERS"], capture_output=True, text=True, check=True).stdout
    return pd.read_csv(io.StringIO(out), low_memory=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--mdb", type=Path, help="path to HWSD2.mdb (requires mdbtools)")
    g.add_argument("--layers-csv", type=Path, help="CSV export of the HWSD2_LAYERS table")
    ap.add_argument("--out", type=Path, default=soil.DATA_DIR / "hwsd2.sqlite")
    args = ap.parse_args()

    df = read_from_mdb(args.mdb) if args.mdb else pd.read_csv(args.layers_csv, low_memory=False)
    df.columns = [c.strip().upper() for c in df.columns]
    print(f"Read {len(df):,} rows, {len(df.columns)} columns.")

    missing_required = [c for c in REQUIRED if c not in df.columns]
    if missing_required:
        sys.exit(f"MISSING required columns: {missing_required}\nColumns present: {sorted(df.columns)}")
    if "SMU" not in "".join(df.columns) and "HWSD2_SMU_ID" not in df.columns:
        sys.exit("No HWSD2_SMU_ID column: is this the HWSD2_LAYERS table?")

    layers = sorted(df["LAYER"].astype(str).str.upper().unique())
    print("Layers found:", layers, "(expected D1..D7)")
    print("\nNumeric properties used by FarmScan:")
    for col, key in soil.NUMERIC_PROPS.items():
        print(f"  {'ok     ' if col in df.columns else 'MISSING'} {col:<12s} -> {key}")
    print("Categorical:", {c: (c in df.columns) for c in CATEGORICAL})

    df["LAYER"] = df["LAYER"].astype(str).str.upper()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        args.out.unlink()
    conn = sqlite3.connect(args.out)
    df.to_sql("HWSD2_LAYERS", conn, index=False)
    conn.execute("CREATE INDEX idx_layers_smu ON HWSD2_LAYERS (HWSD2_SMU_ID, LAYER)")
    conn.commit()
    conn.close()
    print(f"\nWrote {args.out}  ({args.out.stat().st_size / 1e6:.1f} MB)")

    st = soil.status()
    print("\nRaster:", "found" if st["raster_found"] else f"NOT FOUND at {st['raster']} -- copy HWSD2.bil (+ .hdr/.prj) there")
    print("Ready." if st["ready"] else "Not ready yet.")


if __name__ == "__main__":
    main()
