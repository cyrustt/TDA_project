"""
verify_feature_pipeline.py
--------------------------
Light-weight verification script for acoustic-only pipelines.

This supports multiple run prefixes, e.g.:

    out_1000/acoustic.parquet
    out_1000/acoustic_umap.parquet

    out_10000/acoustic.parquet
    out_10000/acoustic_umap.parquet

You pass the prefix via environment variable:

    export OUTDIR=out_1000
    python verify_feature_pipeline.py
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import json
import sys

# ---------------- CONFIG ----------------
OUTDIR = Path(os.environ.get("OUTDIR", "out"))
ACOUSTIC_FILE = OUTDIR / "acoustic.parquet"
UMAP_FILE      = OUTDIR / "acoustic_umap.parquet"
TDA_FILE       = OUTDIR / "tda_summary.json"  # optional


NUMERIC_COLS = [
    'tempo','loudness','time_sig','key',
    'mode','duration','danceability','energy'
]

# ---------------- UTIL ----------------
def ok(cond, msg):
    if cond:
        print(f"✔ {msg}")
    else:
        raise AssertionError(f"✘ {msg}")


# ---------------- MAIN ----------------
def main():
    print(f"=== VERIFYING OUTPUT IN: {OUTDIR} ===")

    # ---------- 1) Acoustic table ----------
    ok(ACOUSTIC_FILE.exists(), f"acoustic.parquet present -> {ACOUSTIC_FILE}")
    df = pd.read_parquet(ACOUSTIC_FILE)
    ok(len(df) > 0, "acoustic rows > 0")

    # Required columns
    ok(all(c in df.columns for c in NUMERIC_COLS), "all required acoustic columns present")

    # No NaNs
    ok(df[NUMERIC_COLS].notna().all().all(), "no NaNs in numeric feature columns")

    # Song IDs
    ok("song_id" in df.columns, "song_id present")
    ok(df["song_id"].is_unique, "song_id unique")

    # Standardization checks
    means = df[NUMERIC_COLS].mean().abs()
    stds  = df[NUMERIC_COLS].std()

    ok((means < 0.25).all(), "standardized means ~ 0 (|mean| < 0.25)")

    # If any column is constant, std≈0 is allowed.
    nonconstant = stds > 0.05
    ok((stds[nonconstant].sub(1).abs() < 0.25).all(),
       "standardized std ~ 1 on nonconstant columns")

    const_cols = stds[stds <= 0.05].index.tolist()
    if const_cols:
        print(f"ℹ Constant or near-constant columns: {const_cols}")

    # ---------- 2) UMAP embedding ----------
    ok(UMAP_FILE.exists(), f"acoustic_umap.parquet present -> {UMAP_FILE}")
    umap = pd.read_parquet(UMAP_FILE)

    ok({"x","y","cluster","song_id"}.issubset(umap.columns),
       "UMAP columns present")

    ok(len(umap) == len(df),
       "UMAP rows match acoustic rows")

    ok(set(umap["song_id"]).issubset(set(df["song_id"])),
       "UMAP song_ids subset of acoustic table")

    # ---------- 3) TDA summary (optional) ----------
    if TDA_FILE.exists():
        with open(TDA_FILE,"r") as f:
            js = json.load(f)

        ok("bottleneck_H0" in js, "TDA summary contains bottleneck_H0")
        ok("bottleneck_H1" in js, "TDA summary contains bottleneck_H1")
    else:
        print("ℹ No tda_summary.json found (skipping TDA checks).")

    print(f"\n=== ALL CHECKS PASSED for {OUTDIR} ===")


if __name__ == "__main__":
    main()