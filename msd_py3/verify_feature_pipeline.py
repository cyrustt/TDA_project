# verify_feature_pipeline.py
import os, json, random, glob
import numpy as np
import pandas as pd

# --- Config (match feature_pipeline) ---
OUTDIR = "./out"
ACOUSTIC_FILE = f"{OUTDIR}/acoustic.parquet"
UMAP_FILE     = f"{OUTDIR}/acoustic_umap.parquet"
EDGES_FILE    = f"{OUTDIR}/behavior_edges.parquet"   # may not exist if you skipped triplets
TDA_JSON      = f"{OUTDIR}/tda_summary.json"

REQUIRED_ACOUSTIC_COLS = [
    "song_id","title","artist",
    "tempo","loudness","time_sig","key","mode","duration","danceability","energy"
]
NUMERIC_COLS = ["tempo","loudness","time_sig","key","mode","duration","danceability","energy"]

def ok(cond, msg):
    if cond:  print(f"✔ {msg}")
    else:     raise AssertionError(f"✘ {msg}")

def exists(p, msg):
    ok(os.path.isfile(p), f"{msg}: found -> {p}")

def in_range(series, lo=None, hi=None, allowed=None, name="col"):
    s = series.dropna()
    if allowed is not None:
        bad = ~s.isin(allowed)
        ok((~bad).all(), f"{name} values valid (allowed={allowed})")
        return
    if lo is not None: ok((s >= lo).all(), f"{name} >= {lo}")
    if hi is not None: ok((s <= hi).all(), f"{name} <= {hi}")

def main():
    # 0) Files exist
    exists(ACOUSTIC_FILE, "acoustic feature table")
    df = pd.read_parquet(ACOUSTIC_FILE)
    ok(len(df) > 0, f"acoustic rows > 0  (n={len(df)})")
    ok(set(REQUIRED_ACOUSTIC_COLS).issubset(df.columns), "all required acoustic columns present")

    # 1) Schema/NaNs
    ok(df[NUMERIC_COLS].notna().all().all(), "no NaNs in numeric feature columns")
    ok(df["song_id"].notna().all(), "song_id present for all rows")
    ok(df["song_id"].is_unique, "song_id unique in acoustic table")

    # 2) Value sanity (after standardization, means near 0, stdev near 1)
    # If you changed scaling, loosen/remove these checks.
    means = df[NUMERIC_COLS].mean().abs()


    vals = df[NUMERIC_COLS].to_numpy()

    # Population std to match StandardScaler
    std0 = vals.std(axis=0, ddof=0)

    # Treat columns with (nearly) zero variance as "constant" and OK
    CONST_EPS = 1e-12
    is_const = std0 < CONST_EPS

    # Only enforce std≈1 on non-constant columns
    std_err = np.abs(std0 - 1.0)

    ok((std_err[~is_const] < 0.25).all(),
    "standardized std ~ 1 on non-constant columns (|std-1| < 0.25)")

    if is_const.any():
        const_cols = [NUMERIC_COLS[i] for i, c in enumerate(is_const) if c]
        print(f"ℹ Constant or near-constant columns (std≈0): {const_cols}")


    ok((means < 0.25).all(), "standardized means ~ 0 (|mean| < 0.25)")
    

    # 3) Raw plausibility checks from unscaled info you might still have:
    #    If you skipped scaling, comment out the block above and use these tighter bounds instead.
    # in_range(df["time_sig"], allowed=set([0,1,2,3,4,5,6,7,8,9,10]), name="time_sig")
    # in_range(df["key"],      allowed=set(range(12)), name="key")
    # in_range(df["mode"],     allowed=set([0,1]),     name="mode")

    # 4) UMAP output
    exists(UMAP_FILE, "UMAP embedding")
    emb = pd.read_parquet(UMAP_FILE)
    ok({"x","y","cluster","song_id"}.issubset(emb.columns), "UMAP columns present")
    ok(len(emb) == len(df), f"UMAP rows match acoustic rows (n={len(emb)})")
    ok(emb["song_id"].isin(df["song_id"]).all(), "UMAP song_ids are subset of acoustic table")

    # 5) Behavioral graph & TDA (optional if triplets present)
    if os.path.isfile(EDGES_FILE):
        edf = pd.read_parquet(EDGES_FILE)
        ok({"a","b","w"}.issubset(edf.columns), "behavioral edges columns present")
        ok(len(edf) > 0, f"behavioral edges > 0 (m={len(edf)})")
        ok((edf["w"] >= 2).all(), "edge weights >= 2 (by construction)")

    if os.path.isfile(TDA_JSON):
        with open(TDA_JSON) as f: tda = json.load(f)
        ok("bottleneck_H0" in tda, "TDA summary contains bottleneck_H0")
        # H1 may be NaN if not available; just ensure key present
        ok("bottleneck_H1" in tda, "TDA summary contains bottleneck_H1")

    # 6) Spot-check consistency for a few random songs:
    sample = df.sample(n=min(5, len(df)), random_state=42)
    ok(sample["song_id"].isin(emb["song_id"]).all(), "random sample present in UMAP")
    if os.path.isfile(EDGES_FILE):
        ids = set(sample["song_id"])
        # If there are edges, at least some sampled ids should appear in edges
        appears = pd.read_parquet(EDGES_FILE)
        ok((appears["a"].isin(ids) | appears["b"].isin(ids)).any(),
           "random sample appears in behavioral edges (if edges exist)")

    print("\nAll checks passed 🎉")

if __name__ == "__main__":
    main()
