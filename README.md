# Topological & Acoustic Structure of the Million Song Dataset

A reproducible pipeline for clustering, co-listening graph construction, and persistent homology.

This repository implements a complete workflow for analyzing the Million Song Dataset (MSD) using:
- Acoustic feature extraction (from MSD HDF5 summary files)
- UMAP for nonlinear dimensionality reduction
- HDBSCAN for density-based clustering
- Co-listening behavioral graph from Taste Profile triplets
- Topological Data Analysis (TDA) via persistent homology
- Visualization notebooks comparing acoustic vs behavioral structure

Everything is designed to be fully reproducible, scalable, and friendly for graders/researchers.


## 1. Requirements

Create a virtual environment:
```python
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

You will need two datasets:

(A) MSD Summary File (≈300MB)

Download:
- Official: http://millionsongdataset.com/sites/default/files/AdditionalFiles/msd_summary_file.h5
- Or use the Google Drive copy: https://drive.google.com/file/d/1VWZMnqdtjZvQY5GC0_Sn3ZD_I1V2LbBU/view?usp=sharing

(B) Taste Profile: train_triplets.txt (≈500MB)

Download:
- Official: http://labrosa.ee.columbia.edu/~dpwe/tmp/train_triplets.txt.zip
- Or use the Google Drive copy: https://drive.google.com/file/d/1VWZMnqdtjZvQY5GC0_Sn3ZD_I1V2LbBU/view?usp=sharing

## 2. File Structure
This is what the file structure should look like, after downloading the required datasets.

```
msd_py3/
    feature_pipeline.py       # main pipeline
    hdf5_getters.py           # official MSD helper functions
out_XXXX/
    acoustic_XXXX.parquet
    acoustic_umap_XXXX.parquet
    behavior_edges_XXXX.parquet
    ...
out_
	artist_terms_.parquet	  # contains the artists for each song
1000_analysis.ipynb
50000_analysis.ipynb
msd_summary_file.h5
train_triplets.txt
train_triplets_small.txt
```

## 3. Quick Start (1000-song test subset)

This is the recommended way to test the pipeline end-to-end.
```python
export MSD_SUMMARY_FILE="./msd_summary_file.h5"
export TASTE_TRIPLETS="./train_triplets.txt"
export SAMPLE_SONGS=1000

python msd_py3/feature_pipeline.py
```

This generates a directory:
```
out_1000/
    acoustic_1000.parquet
    acoustic_umap_1000.parquet
    behavior_edges_1000.parquet
    D_ac_1000.npy
    D_be_1000.npy
    dgms_ac_1000.npy
    tda_subset_1000.parquet
    tda_summary_1000.json
```

## 4. Full Pipeline (Large scale, 50k–100k songs)
```python
export MSD_SUMMARY_FILE="./msd_summary_file.h5"
export TASTE_TRIPLETS="./train_triplets.txt"
export SAMPLE_SONGS=50000

python msd_py3/feature_pipeline.py
```

This produces:
```
out_50000/
    acoustic_50000.parquet
    acoustic_umap_50000.parquet
    behavior_edges_50000.parquet
    TDA files...
```
Everything is identical except for dataset size.


## 5. Notebooks
```
1000_analysis.ipynb
50000_analysis.ipynb
```

Includes:
- UMAP embedding
- HDBSCAN clusters
- Representative artists
- Behavioral graph visualization
- Persistent diagrams (acoustic vs behavioral)


## 6. Pipeline Overview

The full processing pipeline consists of the following steps:

1. **Sampling**  
   Randomly sample `SAMPLE_SONGS` tracks from the MSD summary file for scalable experimentation.

2. **Acoustic Feature Extraction**  
   Extract tempo, loudness, key, mode, duration, danceability, and energy from the MSD summary file, then apply **z-score normalization** across all numerical features.

3. **UMAP Embedding**  
   Compute a 2D nonlinear embedding from the standardized acoustic feature space to visualize global and local similarity.

4. **HDBSCAN Clustering**  
   Apply density-based clustering (no need to pre-specify the number of clusters). Outputs robust clusters and a “noise” group (cluster = –1).

5. **Co-listening Graph Construction**  
   Using the Taste Profile subset:  
   Songs `A` and `B` are connected if **≥ 2 users** listened to both.  
   This produces a weighted behavioral graph encoding user-similarity structure.

6. **Topological Data Analysis (TDA)**  
   - Build **Vietoris–Rips persistence diagrams** for both the acoustic and behavioral distance spaces.  
   - Compute **bottleneck distances** to quantify structural similarity between the two modalities.  
   - Save finite H₀ (connected components) and H₁ (cycles/loops) summaries.

7. **Visualization**  
   - UMAP cluster plot  
   - Cluster-level genre & artist summaries  
   - Behavioral co-listening graph (optionally **pruned** and **sparsified**) for interpretability


_Acknowledgement: This document was partially drafted and revised with the assistance of ChatGPT (OpenAI, 2025) to improve clarity and organization._