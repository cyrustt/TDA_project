# msd_py3/extract_genres_summary.py
# ----------------------------------
# Extract (song_id, artist_name, genre) from the MSD SUMMARY FILE

import os
import sys
import pandas as pd
import numpy as np
import h5py

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

MSD_SUMMARY_FILE = os.environ.get("MSD_SUMMARY_FILE", "./MSD_summary_file.h5")
TAG = os.environ.get("TAG", "")
OUT_PATH = os.path.join(PROJECT_ROOT, f"out_{TAG}", f"artist_terms_{TAG}.parquet")

ACOUSTIC_FILE = os.environ.get("ACOUSTIC_FILE", None)

def decode(x):
    if isinstance(x, (bytes, bytearray, np.bytes_)):
        return x.decode("utf-8", "ignore")
    return str(x)

def load_song_filter():
    if ACOUSTIC_FILE is None or not os.path.isfile(ACOUSTIC_FILE):
        return None
    df = pd.read_parquet(ACOUSTIC_FILE)
    return set(df["song_id"].astype(str).tolist())

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
        n = meta.shape[0]

        print(f"[INFO] Summary contains {n} songs.")

        for i in range(n):

            sid = decode(meta[i]["song_id"])

            if keep_songs is not None and sid not in keep_songs:
                continue

            artist_name = decode(meta[i]["artist_name"])
            genre = decode(meta[i]["genre"])   # <- THE ONLY GENRE FIELD

            rows.append((sid, artist_name, genre))

    df = pd.DataFrame(rows, columns=["song_id", "artist_name", "genre"])
    df.to_parquet(OUT_PATH, index=False)

    print(f"[INFO] Rows written: {len(df)}")
    print(f"✅ Saved -> {OUT_PATH}")

if __name__ == "__main__":
    main()