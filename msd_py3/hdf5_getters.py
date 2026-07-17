# hdf5_getters.py: Python 3 / PyTables 3 compatible
# Minimal, self-contained reimplementation of the getters used in:
#   - create_aggregate_file.py / hdf5_utils.py
#   - feature_pipeline.py
# for Million Song Dataset-style HDF5 files.

import numpy as np
import tables

# -------------------- open helpers --------------------

def open_h5_file_read(h5filename):
    """Open an existing H5 file in read mode."""
    return tables.open_file(h5filename, mode='r')

def open_h5_file_append(h5filename):
    """Open an existing H5 file in append mode."""
    return tables.open_file(h5filename, mode='a')

def get_num_songs(h5):
    """Number of songs (rows) stored in this file (aggregate files may have >1)."""
    return h5.root.metadata.songs.nrows

# -------------------- internal helpers --------------------

def _row(h5, group_name, table_name, songidx):
    tbl = getattr(getattr(h5.root, group_name), table_name)
    if songidx < 0 or songidx >= tbl.nrows:
        raise IndexError("songidx out of range for table %s/%s" % (group_name, table_name))
    return tbl[songidx]

def _col(h5, group_name, table_name, col, songidx):
    return _row(h5, group_name, table_name, songidx)[col]

def _maybe_bytes(x):
    # many string columns are fixed-width bytes; keep bytes to match original API
    return x

def _array1d(h5, group_name, array_name, start_idx, end_idx=None):
    arr = getattr(getattr(h5.root, group_name), array_name)
    if end_idx is None:
        return np.array(arr[start_idx:])
    return np.array(arr[start_idx:end_idx])

def _array2d(h5, group_name, array_name, start_idx, length, width):
    arr = getattr(getattr(h5.root, group_name), array_name)
    return np.array(arr[start_idx:start_idx+length]).reshape((length, width))

# -------------------- METADATA (per-song row) --------------------

def get_artist_familiarity(h5, songidx=0):   return float(_col(h5,'metadata','songs','artist_familiarity',songidx))
def get_artist_hotttnesss(h5, songidx=0):    return float(_col(h5,'metadata','songs','artist_hotttnesss',songidx))
def get_artist_id(h5, songidx=0):            return _maybe_bytes(_col(h5,'metadata','songs','artist_id',songidx))
def get_artist_mbid(h5, songidx=0):          return _maybe_bytes(_col(h5,'metadata','songs','artist_mbid',songidx))
def get_artist_playmeid(h5, songidx=0):      return int(_col(h5,'metadata','songs','artist_playmeid',songidx))
def get_artist_7digitalid(h5, songidx=0):    return int(_col(h5,'metadata','songs','artist_7digitalid',songidx))
def get_artist_latitude(h5, songidx=0):      return float(_col(h5,'metadata','songs','artist_latitude',songidx))
def get_artist_location(h5, songidx=0):      return _maybe_bytes(_col(h5,'metadata','songs','artist_location',songidx))
def get_artist_longitude(h5, songidx=0):     return float(_col(h5,'metadata','songs','artist_longitude',songidx))
def get_artist_name(h5, songidx=0):          return _maybe_bytes(_col(h5,'metadata','songs','artist_name',songidx))
def get_release(h5, songidx=0):              return _maybe_bytes(_col(h5,'metadata','songs','release',songidx))
def get_release_7digitalid(h5, songidx=0):   return int(_col(h5,'metadata','songs','release_7digitalid',songidx))
def get_song_id(h5, songidx=0):              return _maybe_bytes(_col(h5,'metadata','songs','song_id',songidx))
def get_song_hotttnesss(h5, songidx=0):      return float(_col(h5,'metadata','songs','song_hotttnesss',songidx))
def get_title(h5, songidx=0):                return _maybe_bytes(_col(h5,'metadata','songs','title',songidx))
def get_track_7digitalid(h5, songidx=0):     return int(_col(h5,'metadata','songs','track_7digitalid',songidx))

# Array indices for metadata arrays
def _idx_similar_artists(h5, songidx=0): return int(_col(h5,'metadata','songs','idx_similar_artists',songidx))
def _idx_artist_terms(h5, songidx=0):    return int(_col(h5,'metadata','songs','idx_artist_terms',songidx))

# Metadata arrays
def get_similar_artists(h5, songidx=0):
    start = _idx_similar_artists(h5, songidx)
    # next row's idx (or end)
    n = get_num_songs(h5)
    end = _idx_similar_artists(h5, songidx+1) if songidx+1 < n else getattr(h5.root.metadata, 'similar_artists').shape[0]
    return _array1d(h5, 'metadata', 'similar_artists', start, end)

def get_artist_terms(h5, songidx=0):
    start = _idx_artist_terms(h5, songidx)
    n = get_num_songs(h5)
    end = _idx_artist_terms(h5, songidx+1) if songidx+1 < n else getattr(h5.root.metadata, 'artist_terms').shape[0]
    return _array1d(h5, 'metadata', 'artist_terms', start, end)

def get_artist_terms_freq(h5, songidx=0):
    start = _idx_artist_terms(h5, songidx)
    n = get_num_songs(h5)
    end = _idx_artist_terms(h5, songidx+1) if songidx+1 < n else getattr(h5.root.metadata, 'artist_terms_freq').shape[0]
    return _array1d(h5, 'metadata', 'artist_terms_freq', start, end)

def get_artist_terms_weight(h5, songidx=0):
    start = _idx_artist_terms(h5, songidx)
    n = get_num_songs(h5)
    end = _idx_artist_terms(h5, songidx+1) if songidx+1 < n else getattr(h5.root.metadata, 'artist_terms_weight').shape[0]
    return _array1d(h5, 'metadata', 'artist_terms_weight', start, end)

# -------------------- ANALYSIS (per-song row) --------------------

def get_analysis_sample_rate(h5, songidx=0):      return float(_col(h5,'analysis','songs','analysis_sample_rate',songidx))
def get_audio_md5(h5, songidx=0):                 return _maybe_bytes(_col(h5,'analysis','songs','audio_md5',songidx))
def get_danceability(h5, songidx=0):              return float(_col(h5,'analysis','songs','danceability',songidx))
def get_duration(h5, songidx=0):                  return float(_col(h5,'analysis','songs','duration',songidx))
def get_end_of_fade_in(h5, songidx=0):            return float(_col(h5,'analysis','songs','end_of_fade_in',songidx))
def get_energy(h5, songidx=0):                    return float(_col(h5,'analysis','songs','energy',songidx))
def get_key(h5, songidx=0):                       return int(_col(h5,'analysis','songs','key',songidx))
def get_key_confidence(h5, songidx=0):            return float(_col(h5,'analysis','songs','key_confidence',songidx))
def get_loudness(h5, songidx=0):                  return float(_col(h5,'analysis','songs','loudness',songidx))
def get_mode(h5, songidx=0):                      return int(_col(h5,'analysis','songs','mode',songidx))
def get_mode_confidence(h5, songidx=0):           return float(_col(h5,'analysis','songs','mode_confidence',songidx))
def get_start_of_fade_out(h5, songidx=0):         return float(_col(h5,'analysis','songs','start_of_fade_out',songidx))
def get_tempo(h5, songidx=0):                     return float(_col(h5,'analysis','songs','tempo',songidx))
def get_time_signature(h5, songidx=0):            return int(_col(h5,'analysis','songs','time_signature',songidx))
def get_time_signature_confidence(h5, songidx=0): return float(_col(h5,'analysis','songs','time_signature_confidence',songidx))
def get_track_id(h5, songidx=0):                  return _maybe_bytes(_col(h5,'analysis','songs','track_id',songidx))

# Array indices for analysis arrays
def _idx(h5, name, songidx):
    return int(_col(h5,'analysis','songs',name,songidx))

def _idx_segments_start(h5, songidx=0):        return _idx(h5,'idx_segments_start',songidx)
def _idx_segments_confidence(h5, songidx=0):   return _idx(h5,'idx_segments_confidence',songidx)
def _idx_segments_pitches(h5, songidx=0):      return _idx(h5,'idx_segments_pitches',songidx)
def _idx_segments_timbre(h5, songidx=0):       return _idx(h5,'idx_segments_timbre',songidx)
def _idx_segments_loudness_max(h5, songidx=0): return _idx(h5,'idx_segments_loudness_max',songidx)
def _idx_segments_loudness_max_time(h5,songidx=0): return _idx(h5,'idx_segments_loudness_max_time',songidx)
def _idx_segments_loudness_start(h5,songidx=0): return _idx(h5,'idx_segments_loudness_start',songidx)
def _idx_sections_start(h5, songidx=0):        return _idx(h5,'idx_sections_start',songidx)
def _idx_sections_confidence(h5, songidx=0):   return _idx(h5,'idx_sections_confidence',songidx)
def _idx_beats_start(h5, songidx=0):           return _idx(h5,'idx_beats_start',songidx)
def _idx_beats_confidence(h5, songidx=0):      return _idx(h5,'idx_beats_confidence',songidx)
def _idx_bars_start(h5, songidx=0):            return _idx(h5,'idx_bars_start',songidx)
def _idx_bars_confidence(h5, songidx=0):       return _idx(h5,'idx_bars_confidence',songidx)
def _idx_tatums_start(h5, songidx=0):          return _idx(h5,'idx_tatums_start',songidx)
def _idx_tatums_confidence(h5, songidx=0):     return _idx(h5,'idx_tatums_confidence',songidx)

# For each array, compute slice [start:end] using next row's idx or array length

def _slice_bounds(h5, group, array, idx_fn, songidx):
    start = idx_fn(h5, songidx)
    n = get_num_songs(h5)
    if songidx + 1 < n:
        end = idx_fn(h5, songidx + 1)
    else:
        end = getattr(getattr(h5.root, group), array).shape[0]
    return start, end

# segments
def get_segments_start(h5, songidx=0):
    s,e = _slice_bounds(h5,'analysis','segments_start',_idx_segments_start,songidx)
    return _array1d(h5,'analysis','segments_start',s,e)

def get_segments_confidence(h5, songidx=0):
    s,e = _slice_bounds(h5,'analysis','segments_confidence',_idx_segments_confidence,songidx)
    return _array1d(h5,'analysis','segments_confidence',s,e)

def get_segments_pitches(h5, songidx=0):
    s,e = _slice_bounds(h5,'analysis','segments_pitches',_idx_segments_pitches,songidx)
    # segments_pitches stored as (N,12)
    length = e - s
    if length <= 0: return np.zeros((0,12), dtype=np.float64)
    arr = getattr(h5.root.analysis, 'segments_pitches')
    return np.array(arr[s:e]).reshape((length,12))

def get_segments_timbre(h5, songidx=0):
    s,e = _slice_bounds(h5,'analysis','segments_timbre',_idx_segments_timbre,songidx)
    length = e - s
    if length <= 0: return np.zeros((0,12), dtype=np.float64)
    arr = getattr(h5.root.analysis, 'segments_timbre')
    return np.array(arr[s:e]).reshape((length,12))

def get_segments_loudness_max(h5, songidx=0):
    s,e = _slice_bounds(h5,'analysis','segments_loudness_max',_idx_segments_loudness_max,songidx)
    return _array1d(h5,'analysis','segments_loudness_max',s,e)

def get_segments_loudness_max_time(h5, songidx=0):
    s,e = _slice_bounds(h5,'analysis','segments_loudness_max_time',_idx_segments_loudness_max_time,songidx)
    return _array1d(h5,'analysis','segments_loudness_max_time',s,e)

def get_segments_loudness_start(h5, songidx=0):
    s,e = _slice_bounds(h5,'analysis','segments_loudness_start',_idx_segments_loudness_start,songidx)
    return _array1d(h5,'analysis','segments_loudness_start',s,e)

# sections
def get_sections_start(h5, songidx=0):
    s,e = _slice_bounds(h5,'analysis','sections_start',_idx_sections_start,songidx)
    return _array1d(h5,'analysis','sections_start',s,e)

def get_sections_confidence(h5, songidx=0):
    s,e = _slice_bounds(h5,'analysis','sections_confidence',_idx_sections_confidence,songidx)
    return _array1d(h5,'analysis','sections_confidence',s,e)

# beats
def get_beats_start(h5, songidx=0):
    s,e = _slice_bounds(h5,'analysis','beats_start',_idx_beats_start,songidx)
    return _array1d(h5,'analysis','beats_start',s,e)

def get_beats_confidence(h5, songidx=0):
    s,e = _slice_bounds(h5,'analysis','beats_confidence',_idx_beats_confidence,songidx)
    return _array1d(h5,'analysis','beats_confidence',s,e)

# bars
def get_bars_start(h5, songidx=0):
    s,e = _slice_bounds(h5,'analysis','bars_start',_idx_bars_start,songidx)
    return _array1d(h5,'analysis','bars_start',s,e)

def get_bars_confidence(h5, songidx=0):
    s,e = _slice_bounds(h5,'analysis','bars_confidence',_idx_bars_confidence,songidx)
    return _array1d(h5,'analysis','bars_confidence',s,e)

# tatums
def get_tatums_start(h5, songidx=0):
    s,e = _slice_bounds(h5,'analysis','tatums_start',_idx_tatums_start,songidx)
    return _array1d(h5,'analysis','tatums_start',s,e)

def get_tatums_confidence(h5, songidx=0):
    s,e = _slice_bounds(h5,'analysis','tatums_confidence',_idx_tatums_confidence,songidx)
    return _array1d(h5,'analysis','tatums_confidence',s,e)

# -------------------- MUSICBRAINZ (per-song row + arrays) --------------------

def get_year(h5, songidx=0):                    return int(_col(h5,'musicbrainz','songs','year',songidx))

def _idx_artist_mbtags(h5, songidx=0):          return int(_col(h5,'musicbrainz','songs','idx_artist_mbtags',songidx))

def get_artist_mbtags(h5, songidx=0):
    start = _idx_artist_mbtags(h5, songidx)
    n = get_num_songs(h5)
    end = _idx_artist_mbtags(h5, songidx+1) if songidx+1 < n else getattr(h5.root.musicbrainz, 'artist_mbtags').shape[0]
    return _array1d(h5,'musicbrainz','artist_mbtags',start,end)

def get_artist_mbtags_count(h5, songidx=0):
    start = _idx_artist_mbtags(h5, songidx)
    n = get_num_songs(h5)
    end = _idx_artist_mbtags(h5, songidx+1) if songidx+1 < n else getattr(h5.root.musicbrainz, 'artist_mbtags_count').shape[0]
    return _array1d(h5,'musicbrainz','artist_mbtags_count',start,end)
