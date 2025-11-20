# hdf5_utils.py  (Python 3 + PyTables 3 compatible)
import os, sys, numpy as np, tables
import hdf5_descriptors as DESC
from hdf5_getters import *

ARRAY_DESC_SIMILAR_ARTISTS = 'array of similar artists Echo Nest id'
ARRAY_DESC_ARTIST_TERMS = 'array of terms (Echo Nest tags) for an artist'
ARRAY_DESC_ARTIST_TERMS_FREQ = 'array of term (Echo Nest tags) frequencies for an artist'
ARRAY_DESC_ARTIST_TERMS_WEIGHT = 'array of term (Echo Nest tags) weights for an artist'
ARRAY_DESC_SEGMENTS_START = 'array of start times of segments'
ARRAY_DESC_SEGMENTS_CONFIDENCE = 'array of confidence of segments'
ARRAY_DESC_SEGMENTS_PITCHES = 'array of pitches of segments (chromas)'
ARRAY_DESC_SEGMENTS_TIMBRE = 'array of timbre of segments (MFCC-like)'
ARRAY_DESC_SEGMENTS_LOUDNESS_MAX = 'array of max loudness of segments'
ARRAY_DESC_SEGMENTS_LOUDNESS_MAX_TIME = 'array of max loudness time of segments'
ARRAY_DESC_SEGMENTS_LOUDNESS_START = 'array of loudness of segments at start time'
ARRAY_DESC_SECTIONS_START = 'array of start times of sections'
ARRAY_DESC_SECTIONS_CONFIDENCE = 'array of confidence of sections'
ARRAY_DESC_BEATS_START = 'array of start times of beats'
ARRAY_DESC_BEATS_CONFIDENCE = 'array of confidence of beats'
ARRAY_DESC_BARS_START = 'array of start times of bars'
ARRAY_DESC_BARS_CONFIDENCE = 'array of confidence of bars'
ARRAY_DESC_TATUMS_START = 'array of start times of tatums'
ARRAY_DESC_TATUMS_CONFIDENCE = 'array of confidence of tatums'
ARRAY_DESC_ARTIST_MBTAGS = 'array of tags from MusicBrainz for an artist'
ARRAY_DESC_ARTIST_MBTAGS_COUNT = 'array of tag counts from MusicBrainz for an artist'

def create_song_file(h5filename, title='H5 Song File', force=False, complevel=1):
    if (not force) and os.path.exists(h5filename):
        raise ValueError('file exists, cannot create HDF5 song file')
    h5 = tables.open_file(h5filename, mode='w', title=title)
    h5.filters = tables.Filters(complevel=complevel, complib='zlib')
    g = h5.create_group("/", 'metadata', 'metadata about the song')
    t = h5.create_table(g, 'songs', DESC.SongMetaData, 'table of metadata'); t.row.append(); t.flush()
    g = h5.create_group("/", 'analysis', 'Echo Nest analysis of the song')
    t = h5.create_table(g, 'songs', DESC.SongAnalysis, 'table of analysis'); t.row.append(); t.flush()
    g = h5.create_group("/", 'musicbrainz', 'MusicBrainz data')
    t = h5.create_table(g, 'songs', DESC.SongMusicBrainz, 'table of MBZ'); t.row.append(); t.flush()
    create_all_arrays(h5, expectedrows=3)
    h5.close()

def create_aggregate_file(h5filename, title='H5 Aggregate File', force=False, expectedrows=1000, complevel=1, summaryfile=False):
    if (not force) and os.path.exists(h5filename):
        raise ValueError('file exists, cannot create HDF5 song file')
    if summaryfile: title = 'H5 Summary File'
    h5 = tables.open_file(h5filename, mode='w', title=title)
    h5.filters = tables.Filters(complevel=complevel, complib='zlib')
    g = h5.create_group("/", 'metadata', 'metadata about the song')
    h5.create_table(g, 'songs', DESC.SongMetaData, 'metadata table', expectedrows=expectedrows)
    g = h5.create_group("/", 'analysis', 'Echo Nest analysis of the song')
    h5.create_table(g, 'songs', DESC.SongAnalysis, 'analysis table', expectedrows=expectedrows)
    g = h5.create_group("/", 'musicbrainz', 'MusicBrainz data')
    h5.create_table(g, 'songs', DESC.SongMusicBrainz, 'mbz table', expectedrows=expectedrows)
    if not summaryfile: create_all_arrays(h5, expectedrows=expectedrows)
    h5.close()

def create_all_arrays(h5, expectedrows=1000):
    g = h5.root.metadata
    h5.create_earray(g, 'similar_artists', tables.StringAtom(20), (0,), ARRAY_DESC_SIMILAR_ARTISTS)
    h5.create_earray(g, 'artist_terms', tables.StringAtom(256), (0,), ARRAY_DESC_ARTIST_TERMS, expectedrows=expectedrows*40)
    h5.create_earray(g, 'artist_terms_freq', tables.Float64Atom(), (0,), ARRAY_DESC_ARTIST_TERMS_FREQ, expectedrows=expectedrows*40)
    h5.create_earray(g, 'artist_terms_weight', tables.Float64Atom(), (0,), ARRAY_DESC_ARTIST_TERMS_WEIGHT, expectedrows=expectedrows*40)
    g = h5.root.analysis
    h5.create_earray(g, 'segments_start', tables.Float64Atom(), (0,), ARRAY_DESC_SEGMENTS_START)
    h5.create_earray(g, 'segments_confidence', tables.Float64Atom(), (0,), ARRAY_DESC_SEGMENTS_CONFIDENCE, expectedrows=expectedrows*300)
    h5.create_earray(g, 'segments_pitches', tables.Float64Atom(), (0,12), ARRAY_DESC_SEGMENTS_PITCHES, expectedrows=expectedrows*300)
    h5.create_earray(g, 'segments_timbre', tables.Float64Atom(), (0,12), ARRAY_DESC_SEGMENTS_TIMBRE, expectedrows=expectedrows*300)
    h5.create_earray(g, 'segments_loudness_max', tables.Float64Atom(), (0,), ARRAY_DESC_SEGMENTS_LOUDNESS_MAX, expectedrows=expectedrows*300)
    h5.create_earray(g, 'segments_loudness_max_time', tables.Float64Atom(), (0,), ARRAY_DESC_SEGMENTS_LOUDNESS_MAX_TIME, expectedrows=expectedrows*300)
    h5.create_earray(g, 'segments_loudness_start', tables.Float64Atom(), (0,), ARRAY_DESC_SEGMENTS_LOUDNESS_START, expectedrows=expectedrows*300)
    h5.create_earray(g, 'sections_start', tables.Float64Atom(), (0,), ARRAY_DESC_SECTIONS_START, expectedrows=expectedrows*300)
    h5.create_earray(g, 'sections_confidence', tables.Float64Atom(), (0,), ARRAY_DESC_SECTIONS_CONFIDENCE, expectedrows=expectedrows*300)
    h5.create_earray(g, 'beats_start', tables.Float64Atom(), (0,), ARRAY_DESC_BEATS_START, expectedrows=expectedrows*300)
    h5.create_earray(g, 'beats_confidence', tables.Float64Atom(), (0,), ARRAY_DESC_BEATS_CONFIDENCE, expectedrows=expectedrows*300)
    h5.create_earray(g, 'bars_start', tables.Float64Atom(), (0,), ARRAY_DESC_BARS_START, expectedrows=expectedrows*300)
    h5.create_earray(g, 'bars_confidence', tables.Float64Atom(), (0,), ARRAY_DESC_BARS_CONFIDENCE, expectedrows=expectedrows*300)
    h5.create_earray(g, 'tatums_start', tables.Float64Atom(), (0,), ARRAY_DESC_TATUMS_START, expectedrows=expectedrows*300)
    h5.create_earray(g, 'tatums_confidence', tables.Float64Atom(), (0,), ARRAY_DESC_TATUMS_CONFIDENCE, expectedrows=expectedrows*300)
    g = h5.root.musicbrainz
    h5.create_earray(g, 'artist_mbtags', tables.StringAtom(256), (0,), ARRAY_DESC_ARTIST_MBTAGS, expectedrows=expectedrows*5)
    h5.create_earray(g, 'artist_mbtags_count', tables.Int32Atom(), (0,), ARRAY_DESC_ARTIST_MBTAGS_COUNT, expectedrows=expectedrows*5)

def open_h5_file_read(h5filename):   return tables.open_file(h5filename, mode='r')
def open_h5_file_append(h5filename): return tables.open_file(h5filename, mode='a')

def fill_hdf5_aggregate_file(h5, h5_filenames, summaryfile=False):
    counter = 0
    for h5filename in h5_filenames:
        src = open_h5_file_read(h5filename)
        n = get_num_songs(src)
        for i in range(n):
            row = h5.root.metadata.songs.row
            row["artist_familiarity"]   = get_artist_familiarity(src, i)
            row["artist_hotttnesss"]    = get_artist_hotttnesss(src, i)
            row["artist_id"]            = get_artist_id(src, i)
            row["artist_mbid"]          = get_artist_mbid(src, i)
            row["artist_playmeid"]      = get_artist_playmeid(src, i)
            row["artist_7digitalid"]    = get_artist_7digitalid(src, i)
            row["artist_latitude"]      = get_artist_latitude(src, i)
            row["artist_location"]      = get_artist_location(src, i)
            row["artist_longitude"]     = get_artist_longitude(src, i)
            row["artist_name"]          = get_artist_name(src, i)
            row["release"]              = get_release(src, i)
            row["release_7digitalid"]   = get_release_7digitalid(src, i)
            row["song_id"]              = get_song_id(src, i)
            row["song_hotttnesss"]      = get_song_hotttnesss(src, i)
            row["title"]                = get_title(src, i)
            row["track_7digitalid"]     = get_track_7digitalid(src, i)
            if not summaryfile:
                if counter == 0:
                    row["idx_similar_artists"] = 0
                    row["idx_artist_terms"]    = 0
                else:
                    row["idx_similar_artists"] = h5.root.metadata.similar_artists.shape[0]
                    row["idx_artist_terms"]    = h5.root.metadata.artist_terms.shape[0]
            row.append(); h5.root.metadata.songs.flush()

            if not summaryfile:
                h5.root.metadata.similar_artists.append(get_similar_artists(src, i))
                h5.root.metadata.artist_terms.append(get_artist_terms(src, i))
                h5.root.metadata.artist_terms_freq.append(get_artist_terms_freq(src, i))
                h5.root.metadata.artist_terms_weight.append(get_artist_terms_weight(src, i))

            row = h5.root.analysis.songs.row
            row["analysis_sample_rate"]      = get_analysis_sample_rate(src, i)
            row["audio_md5"]                 = get_audio_md5(src, i)
            row["danceability"]              = get_danceability(src, i)
            row["duration"]                  = get_duration(src, i)
            row["end_of_fade_in"]            = get_end_of_fade_in(src, i)
            row["energy"]                    = get_energy(src, i)
            row["key"]                       = get_key(src, i)
            row["key_confidence"]            = get_key_confidence(src, i)
            row["loudness"]                  = get_loudness(src, i)
            row["mode"]                      = get_mode(src, i)
            row["mode_confidence"]           = get_mode_confidence(src, i)
            row["start_of_fade_out"]         = get_start_of_fade_out(src, i)
            row["tempo"]                     = get_tempo(src, i)
            row["time_signature"]            = get_time_signature(src, i)
            row["time_signature_confidence"] = get_time_signature_confidence(src, i)
            row["track_id"]                  = get_track_id(src, i)

            if not summaryfile:
                a = h5.root.analysis
                if counter == 0:
                    for k in ["idx_segments_start","idx_segments_confidence","idx_segments_pitches",
                              "idx_segments_timbre","idx_segments_loudness_max","idx_segments_loudness_max_time",
                              "idx_segments_loudness_start","idx_sections_start","idx_sections_confidence",
                              "idx_beats_start","idx_beats_confidence","idx_bars_start","idx_bars_confidence",
                              "idx_tatums_start","idx_tatums_confidence"]:
                        row[k] = 0
                else:
                    aroot = h5.root.analysis
                    row["idx_segments_start"]        = aroot.segments_start.shape[0]
                    row["idx_segments_confidence"]   = aroot.segments_confidence.shape[0]
                    row["idx_segments_pitches"]      = aroot.segments_pitches.shape[0]
                    row["idx_segments_timbre"]       = aroot.segments_timbre.shape[0]
                    row["idx_segments_loudness_max"] = aroot.segments_loudness_max.shape[0]
                    row["idx_segments_loudness_max_time"] = aroot.segments_loudness_max_time.shape[0]
                    row["idx_segments_loudness_start"]    = aroot.segments_loudness_start.shape[0]
                    row["idx_sections_start"]        = aroot.sections_start.shape[0]
                    row["idx_sections_confidence"]   = aroot.sections_confidence.shape[0]
                    row["idx_beats_start"]           = aroot.beats_start.shape[0]
                    row["idx_beats_confidence"]      = aroot.beats_confidence.shape[0]
                    row["idx_bars_start"]            = aroot.bars_start.shape[0]
                    row["idx_bars_confidence"]       = aroot.bars_confidence.shape[0]
                    row["idx_tatums_start"]          = aroot.tatums_start.shape[0]
                    row["idx_tatums_confidence"]     = aroot.tatums_confidence.shape[0]

            row.append(); h5.root.analysis.songs.flush()

            if not summaryfile:
                a = h5.root.analysis
                a.segments_start.append(get_segments_start(src, i))
                a.segments_confidence.append(get_segments_confidence(src, i))
                a.segments_pitches.append(get_segments_pitches(src, i))
                a.segments_timbre.append(get_segments_timbre(src, i))
                a.segments_loudness_max.append(get_segments_loudness_max(src, i))
                a.segments_loudness_max_time.append(get_segments_loudness_max_time(src, i))
                a.segments_loudness_start.append(get_segments_loudness_start(src, i))
                a.sections_start.append(get_sections_start(src, i))
                a.sections_confidence.append(get_sections_confidence(src, i))
                a.beats_start.append(get_beats_start(src, i))
                a.beats_confidence.append(get_beats_confidence(src, i))
                a.bars_start.append(get_bars_start(src, i))
                a.bars_confidence.append(get_bars_confidence(src, i))
                a.tatums_start.append(get_tatums_start(src, i))
                a.tatums_confidence.append(get_tatums_confidence(src, i))

            row = h5.root.musicbrainz.songs.row
            row["year"] = get_year(src, i)
            if not summaryfile:
                row["idx_artist_mbtags"] = 0 if counter == 0 else h5.root.musicbrainz.artist_mbtags.shape[0]
            row.append(); h5.root.musicbrainz.songs.flush()
            if not summaryfile:
                mb = h5.root.musicbrainz
                mb.artist_mbtags.append(get_artist_mbtags(src, i))
                mb.artist_mbtags_count.append(get_artist_mbtags_count(src, i))
            counter += 1
        src.close()
