# Topological & Acoustic Structure of the Million Song Dataset

A reproducible pipeline for clustering, co-listening graph construction, and persistent homology.

This repository implements a complete workflow for analyzing the Million Song Dataset (MSD) using:
	•	Acoustic feature extraction (from MSD HDF5 summary files)
	•	UMAP for nonlinear dimensionality reduction
	•	HDBSCAN for density-based clustering
	•	Co-listening behavioral graph from Taste Profile triplets
	•	Topological Data Analysis (TDA) via persistent homology
	•	Visualization notebooks comparing acoustic vs behavioral structure

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
		•	Official: http://millionsongdataset.com/sites/default/files/AdditionalFiles/msd_summary_file.h5
		•	Or use the Google Drive copy included in the instructions

(B) Taste Profile: train_triplets.txt (≈500MB)

	Download:
		•	Official: http://labrosa.ee.columbia.edu/~dpwe/tmp/train_triplets.txt.zip
		•	Or use the included Google Drive link
		•	Or the mini tester file: train_triplets_small.txt

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
export TASTE_TRIPLETS="./train_triplets_small.txt"
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
	•	UMAP embedding
	•	HDBSCAN clusters
	•	Representative artists
	•	Behavioral graph visualization
	•	Persistent diagrams (acoustic vs behavioral)


## 6. Pipeline Overview

The full process is:
	1.	Sampling
Randomly sample SAMPLE_SONGS from the MSD summary file.
	2.	Acoustic Feature Extraction
Extract tempo, loudness, key, mode, duration, danceability, energy, then z-score normalize.
	3.	UMAP Embedding
Compute a 2D embedding from standardized features.
	4.	HDBSCAN Clustering
Find density clusters (no need to pre-specify k).
	5.	Co-listening Graph
From Taste Profile data:
songs A and B connected if ≥2 users listened to both.
	6.	TDA (persistent homology)
	•	Compute Vietoris–Rips persistence diagrams for acoustic and behavioral spaces
	•	Compare diagrams using bottleneck distance
	•	Save H₀ and H₁ topological summaries
	7.	Visualization
	•	UMAP cluster plot
	•	Genre/artist summaries
	•	Behavioral graph (optionally pruned + sparsified)


