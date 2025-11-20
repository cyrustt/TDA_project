# feature_pipeline.py  — fast & robust
import os, json, random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import umap
import hdbscan
from ripser import ripser
from persim import bottleneck
import warnings

# silence persim infinite-death warnings (we pre-filter)
warnings.filterwarnings("ignore", message="dgm.*non-finite death", module="persim.bottleneck")
OUTDIR = Path(os.environ.get('OUTDIR', './out')); OUTDIR.mkdir(parents=True, exist_ok=True)

# ---------- CONFIG ----------
MSD_H5_DIR      = os.environ.get('MSD_H5_DIR', './MillionSongSubset')
TASTE_TRIPLETS  = os.environ.get('TASTE_TRIPLETS', '/path/to/train_triplets.txt')
SAMPLE_SONGS    = int(os.environ.get('SAMPLE_SONGS', '10000'))  # how many songs to extract acoustically
OUTDIR          = Path('./out'); OUTDIR.mkdir(parents=True, exist_ok=True)

# SPEED CONFIG
FAST_MODE     = True
TDA_SUBSAMPLE = int(os.environ.get('TDA_SUBSAMPLE', '1200'))  # subsample for TDA/bottleneck
BN_MAX_POINTS = int(os.environ.get('BN_MAX_POINTS', '800'))   # diag cap for bottleneck

UMAP_N_NEIGHBORS = int(os.environ.get('UMAP_N_NEIGHBORS', '25'))
HDBSCAN_MIN_CLUSTER_SIZE = int(os.environ.get('HDBSCAN_MIN_CLUSTER_SIZE', '50'))



# Which features to collect (must exist in MSD analysis/metadata tables)
NUMERIC_COLS = ['tempo','loudness','time_sig','key','mode','duration','danceability','energy']

# Import getters
from hdf5_getters import (
    open_h5_file_read, get_song_id, get_title, get_artist_name,
    get_tempo, get_loudness, get_time_signature, get_key, get_mode,
    get_duration
)

# ---------- HELPERS ----------
def iter_song_files(root):
    for p, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith('.h5'):
                yield os.path.join(p, f)

def read_song_row(h5path):
    """Return (song_id, title, artist, feature dict) or None on error."""
    try:
        h5 = open_h5_file_read(h5path)
    except Exception:
        return None
    try:
        sid   = get_song_id(h5)
        title = get_title(h5)
        artist= get_artist_name(h5)
        # decode if bytes
        sid   = sid.decode('utf-8','ignore')   if hasattr(sid, 'decode')   else str(sid)
        title = title.decode('utf-8','ignore') if hasattr(title,'decode')  else str(title)
        artist= artist.decode('utf-8','ignore')if hasattr(artist,'decode') else str(artist)

        # danceability/energy may be missing in some old files; try safe access
        row = h5.root.analysis.songs[0]
        dance = float(getattr(row, 'danceability', np.nan))
        energ = float(getattr(row, 'energy', np.nan))

        feat = dict(
            tempo=float(get_tempo(h5)),
            loudness=float(get_loudness(h5)),
            time_sig=int(get_time_signature(h5)),
            key=int(get_key(h5)),
            mode=int(get_mode(h5)),
            duration=float(get_duration(h5)),
            danceability=dance,
            energy=energ,
        )
        return sid, title, artist, feat
    except Exception:
        return None
    finally:
        try: h5.close()
        except Exception: pass

def build_acoustic_table(files, k=SAMPLE_SONGS):
    files = list(files)
    if k and k < len(files):
        random.seed(42)
        files = random.sample(files, k)

    records = []
    for fp in tqdm(files, desc='Extracting acoustic features'):
        row = read_song_row(fp)
        if row is None: 
            continue
        sid, title, artist, feat = row
        # basic sanity on required numeric fields
        if any(np.isnan(feat[c]) for c in ['tempo','loudness','duration']):
            continue
        records.append({
            'song_id': sid, 'title': title, 'artist': artist, **feat
        })

    df = pd.DataFrame(records)
    print(f"[INFO] Collected {len(df)} songs after NaN drop.")
    if df.empty:
        raise RuntimeError("No valid songs collected. Check MSD_H5_DIR and file integrity.")

    # fill optional cols if missing then standardize
    for c in NUMERIC_COLS:
        if c not in df.columns:
            df[c] = 0.0
    df[NUMERIC_COLS] = df[NUMERIC_COLS].astype(float).fillna(0.0)

    scaler = StandardScaler(with_mean=True, with_std=True)
    df[NUMERIC_COLS] = scaler.fit_transform(df[NUMERIC_COLS].to_numpy())

    df.to_parquet(OUTDIR/'acoustic.parquet', index=False)
    return df

def load_triplets(path, keep_ids):
    """Read Taste Profile subset; build co-listen edges (a,b,w) with w>=2."""
    keep = set(keep_ids)
    song_users = {}
    with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
        for line in fh:
            u, s, _plays = line.rstrip('\n').split('\t')
            if s in keep:
                song_users.setdefault(s, set()).add(u)

    # user -> songs played (restricted)
    user_songs = {}
    for s, users in song_users.items():
        for u in users:
            user_songs.setdefault(u, []).append(s)

    # count co-listens
    edge = {}
    for _, songs in user_songs.items():
        ls = songs
        for i in range(len(ls)):
            for j in range(i+1, len(ls)):
                a, b = (ls[i], ls[j]) if ls[i] < ls[j] else (ls[j], ls[i])
                edge[(a,b)] = edge.get((a,b), 0) + 1

    edges = [(a,b,w) for (a,b), w in edge.items() if w >= 5]
    edf = pd.DataFrame(edges, columns=['a','b','w'])
    edf.to_parquet(OUTDIR/'behavior_edges.parquet', index=False)
    return edf

def pairwise_dist(X):
    # Euclidean distance matrix
    n = X.shape[0]
    # (x - y)^2 = x^2 + y^2 - 2 x·y trick
    G = X @ X.T
    diag = np.sum(X*X, axis=1)
    D2 = diag[:,None] + diag[None,:] - 2.0*G
    D2[D2 < 0] = 0.0
    D = np.sqrt(D2, dtype=np.float32)
    return D

def shortest_path_dist(n, edges, weight_col='w'):
    """All-pairs shortest paths on undirected weighted graph; safe fallback."""
    import heapq
    if not edges:
        D = np.ones((n, n), dtype=np.float32); np.fill_diagonal(D, 0.0); return D

    adj = [[] for _ in range(n)]
    for a, b, w in edges:
        d = 1.0 / max(1.0, float(w))
        adj[a].append((b, d)); adj[b].append((a, d))

    INF = 1e12
    D = np.full((n, n), INF, dtype=np.float32)
    for s in range(n):
        D[s,s] = 0.0
        pq = [(0.0, s)]
        while pq:
            d,u = heapq.heappop(pq)
            if d > D[s,u]: 
                continue
            for v,w in adj[u]:
                nd = d + w
                if nd < D[s,v]:
                    D[s,v] = nd
                    heapq.heappush(pq, (nd, v))

    finite_mask = np.isfinite(D)
    diag = np.eye(n, dtype=bool)
    finite_off = D[finite_mask & ~diag]
    if finite_off.size == 0:
        D[:] = 1.0; np.fill_diagonal(D, 0.0)
    else:
        cap = float(finite_off.max())
        D[~finite_mask] = cap
    return D

def vietoris_rips_persistence(D):
    """Return diagrams up to H1 using a distance matrix."""
    r = ripser(D, distance_matrix=True, maxdim=1)
    return r['dgms']

def finite_dgm(dgm):
    if dgm is None: return None
    m = np.isfinite(dgm).all(axis=1)
    return dgm[m]

def trim_diagram(dgm, m):
    if dgm is None or len(dgm) <= m: return dgm
    rng = np.random.default_rng(42)
    idx = rng.choice(len(dgm), size=m, replace=False)
    return dgm[idx]

# ---------- MAIN ----------
def main():
    # 1) Acoustic features
    files = list(iter_song_files(MSD_H5_DIR))
    if not files:
        raise RuntimeError(f"No .h5 files found under {MSD_H5_DIR}")
    print(f"[INFO] Found {len(files)} files under {MSD_H5_DIR}")

    df = build_acoustic_table(files, k=SAMPLE_SONGS)
    df = df.reset_index(drop=True)
    song_to_idx = {s:i for i,s in enumerate(df['song_id'])}

    # FULL acoustic distance (for UMAP)
    X_full = df[NUMERIC_COLS].to_numpy(dtype=np.float32)
    D_ac_full = pairwise_dist(X_full)

    # 2) Behavioral edges (optional)
    edf = None
    idx_edges = []
    if TASTE_TRIPLETS and os.path.isfile(TASTE_TRIPLETS):
        edf = load_triplets(TASTE_TRIPLETS, df['song_id'].tolist())
        edf = edf[edf['a'].isin(song_to_idx) & edf['b'].isin(song_to_idx)]
        idx_edges = [(song_to_idx[a], song_to_idx[b], float(w))
                     for a,b,w in edf[['a','b','w']].to_numpy()]
        print(f"[INFO] Behavioral edges kept: {len(idx_edges)}")
    else:
        print(f"[INFO] Taste Profile file not found at: {TASTE_TRIPLETS}. Skipping behavioral graph.")

    # 3) TDA subsample & distances (this defines D_ac before use)
    if FAST_MODE:
        df_tda = df.sample(n=min(TDA_SUBSAMPLE, len(df)), random_state=42).reset_index(drop=True)
    else:
        df_tda = df

    # map subsample indices
    keep_idx = np.array([song_to_idx[s] for s in df_tda['song_id']], dtype=int)
    X_tda = df_tda[NUMERIC_COLS].to_numpy(dtype=np.float32)

    # acoustic distances for TDA
    D_ac = pairwise_dist(X_tda)

    # behavioral distances for TDA (if any)
    D_be = None
    if idx_edges:
        kept = set(keep_idx.tolist())
        remap = {old:i for i, old in enumerate(keep_idx)}
        sub_edges = [(remap[a], remap[b], w) for a,b,w in idx_edges if a in kept and b in kept]
        if sub_edges:
            D_be = shortest_path_dist(len(df_tda), sub_edges, 'w')
            print(f"[INFO] Behavioral edges on TDA subset: {len(sub_edges)}")
        else:
            print("[INFO] No behavioral edges on the TDA subset; skipping D_be.")
    else:
        print("[INFO] No behavioral edges available; skipping D_be.")

    # 4) Persistence & bottleneck (finite + trimmed)
    dgms_ac = vietoris_rips_persistence(D_ac)
    bn_H0 = None; bn_H1 = None
    if D_be is not None:
        dgms_be = vietoris_rips_persistence(D_be)

        ac_H0 = trim_diagram(finite_dgm(dgms_ac[0]), BN_MAX_POINTS) if len(dgms_ac) > 0 else None
        be_H0 = trim_diagram(finite_dgm(dgms_be[0]), BN_MAX_POINTS) if len(dgms_be) > 0 else None

        ac_H1 = trim_diagram(finite_dgm(dgms_ac[1]), BN_MAX_POINTS) if len(dgms_ac) > 1 else None
        be_H1 = trim_diagram(finite_dgm(dgms_be[1]), BN_MAX_POINTS) if len(dgms_be) > 1 else None

        if ac_H0 is not None and len(ac_H0) and be_H0 is not None and len(be_H0):
            bn_H0 = float(bottleneck(ac_H0, be_H0))
        if ac_H1 is not None and len(ac_H1) and be_H1 is not None and len(be_H1):
            bn_H1 = float(bottleneck(ac_H1, be_H1))
    else:
        print("[INFO] Skipping bottleneck distance (no behavioral space).")

    with open(OUTDIR/'tda_summary.json', 'w') as f:
        json.dump({'bottleneck_H0': bn_H0, 'bottleneck_H1': bn_H1}, f, indent=2)

    # 5) UMAP + HDBSCAN on FULL acoustic set
    reducer = umap.UMAP(
        n_neighbors=25,
        min_dist=0.1,
        metric='euclidean',
        random_state=42
    )

    # full standardized feature matrix (already built above)
    X_full = df[NUMERIC_COLS].to_numpy(dtype=np.float32)

    # 2D acoustic embedding
    emb = reducer.fit_transform(X_full)

    # density-based clustering on the embedding
    HDBSCAN_MIN_CLUSTER_SIZE = 50  # or whatever you want
    clus = hdbscan.HDBSCAN(min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE).fit(emb)

    # save the acoustic UMAP + cluster labels
    pd.DataFrame(
        {
            'x': emb[:, 0],
            'y': emb[:, 1],
            'cluster': clus.labels_,
            'song_id': df['song_id'],
        }
    ).to_parquet(OUTDIR / 'acoustic_umap.parquet', index=False)

    print("Done. Wrote: acoustic.parquet, behavior_edges.parquet (if any), "
          "tda_summary.json, acoustic_umap.parquet")
    
if __name__ == '__main__':
    main()
