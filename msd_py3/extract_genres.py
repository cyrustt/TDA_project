# msd_py3/extract_genres.py
# -------------------------
# Extract (song_id, artist_name, artist_terms[]) from MSD .h5 files
# and save to ./out/artist_terms.parquet

import os
import sys
import pandas as pd

# --- Make sure we can "import msd_py3.*" whether run as a script or module ---
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Prefer the Py3 getters if present
try:
    from msd_py3.hdf5_getters_py3 import (
        open_h5_file_read,
        get_song_id,
        get_artist_name,
        get_artist_terms,
    )
except Exception:
    # fallback to legacy name if your file is named differently
    from msd_py3.hdf5_getters import (  # type: ignore
        open_h5_file_read,
        get_song_id,
        get_artist_name,
        get_artist_terms,
    )

# ---- Config ----
# You can override this path by:  export MSD_H5_DIR="/path/to/MillionSongSubset"
MSD_H5_DIR = os.environ.get("MSD_H5_DIR", os.path.join(PROJECT_ROOT, "MillionSongSubset"))
OUT_PATH = os.path.join(PROJECT_ROOT, "out", "artist_terms.parquet")


def _to_str(x):
    """Decode bytes -> str; pass through str; stringify other types safely."""
    if isinstance(x, (bytes, bytearray)):
        return x.decode("utf-8", "ignore")
    if isinstance(x, str):
        return x
    return str(x)


def collect_terms(msd_dir):
    rows = []
    n_files = 0
    n_rows = 0
    n_terms = 0

    for root, _, files in os.walk(msd_dir):
        for fname in files:
            if not fname.lower().endswith(".h5"):
                continue
            n_files += 1
            path = os.path.join(root, fname)
            try:
                h5 = open_h5_file_read(path)

                sid = _to_str(get_song_id(h5))
                artist = _to_str(get_artist_name(h5))

                terms_arr = get_artist_terms(h5)
                if terms_arr is None:
                    terms = []
                else:
                    # numpy array -> python list, decode each element
                    terms = [_to_str(t) for t in terms_arr.tolist()]

                rows.append((sid, artist, terms))
                n_rows += 1
                n_terms += len(terms)
            except Exception:
                # Skip unreadable/corrupt files quietly
                pass
            finally:
                try:
                    h5.close()
                except Exception:
                    pass

    df = pd.DataFrame(rows, columns=["song_id", "artist_name", "artist_terms"])
    return df, n_files, n_rows, n_terms


def main():
    if not os.path.isdir(MSD_H5_DIR):
        raise SystemExit(f"ERROR: MSD_H5_DIR does not exist: {MSD_H5_DIR}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

    df, n_files, n_rows, n_terms = collect_terms(MSD_H5_DIR)
    df.to_parquet(OUT_PATH, index=False)

    print(f"MSD_H5_DIR: {MSD_H5_DIR}")
    print(f"Scanned .h5 files: {n_files}")
    print(f"Rows written: {n_rows}")
    print(f"Total artist_terms tokens: {n_terms}")
    print(f"✅ Saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
