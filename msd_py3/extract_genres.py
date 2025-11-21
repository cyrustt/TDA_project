# msd_py3/extract_genres_summary.py
# ----------------------------------
# Extract (song_id, artist_name, artist_terms[]) from the MSD SUMMARY FILE

import os
import sys
import pandas as pd
import numpy as np
import h5py

# --- Resolve project root ---
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# ---- ENV VARIABLES ----
MSD_SUMMARY_FILE = os.environ.get("MSD_SUMMARY_FILE", "./MSD_summary_file.h5")
TAG = os.environ.get("TAG", "")
OUT_PATH = os.path.join(PROJECT_ROOT, f"out_{TAG}", f"artist_terms_{TAG}.parquet")

# optional: restrict songs to those used in feature_pipeline
ACOUSTIC_FILE = os.environ.get("ACOUSTIC_FILE", None)


def decode(x):
    """Decode bytes -> str safely."""
    if isinstance(x, (bytes, bytearray)):
        return x.decode("utf-8", "ignore")
    if isinstance(x, np.bytes_):
        return x.astype(str)
    return str(x)


def load_song_filter():
    """Optionally load acoustic file so terms only extracted for those songs."""
    if ACOUSTIC_FILE is None or not os.path.isfile(ACOUSTIC_FILE):
        return None

    df = pd.read_parquet(ACOUSTIC_FILE)
    return set(df["song_id"].tolist())


def main():

    if not os.path.isfile(MSD_SUMMARY_FILE):
        raise SystemExit(f"ERROR: summary file not found: {MSD_SUMMARY_FILE}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    keep_songs = load_song_filter()
    if keep_songs is not None:
        print(f"[INFO] Restricting to {len(keep_songs)} acoustic-subset songs.")

    rows = []

    with h5py.File(MSD_SUMMARY_FILE, "r") as h5:

        meta = h5["metadata"]["songs"]
        artist_terms = h5["metadata"]["artist_terms"]

        n = meta.shape[0]
        print(f"[INFO] Summary contains {n} songs.")

        for i in range(n):

            sid = decode(meta[i]["song_id"])

            if keep_songs is not None and sid not in keep_songs:
                continue

            artist_name = decode(meta[i]["artist_name"])

            # terms: get indices from artist_terms dataset
            term_list = [decode(term) for term in artist_terms[i]]

            rows.append((sid, artist_name, term_list))

    df = pd.DataFrame(rows, columns=["song_id", "artist_name", "artist_terms"])
    df.to_parquet(OUT_PATH, index=False)

    print(f"[INFO] Rows written: {len(df)}")
    print(f"✅ Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()