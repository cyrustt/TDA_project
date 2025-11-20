# Topological & Acoustic Structure of the Million Song Dataset

## A Reproducible Pipeline for Acoustic Clustering, Behavioral Graphs, and TDA Analysis

This repository provides a fully reproducible pipeline for analyzing the Million Song Dataset (MSD) using:
	•	Acoustic feature extraction from HDF5 files
	•	UMAP nonlinear embedding
	•	HDBSCAN unsupervised clustering
	•	Co-listening graph construction from the Taste Profile subset
	•	Topological Data Analysis (TDA) using persistent homology
	•	Visualization & interpretation notebooks

The project is designed so anyone (e.g., instructors, graders, other researchers) can:
	1.	Run the full pipeline on the official MSD dataset
	2.	Test the code using the provided 1000-song mini subset
	3.	Scale up to 50k or more songs by changing one environment variable

⸻

Repository Structure

TDA_project/
│
├── msd_py3/                    # Feature pipeline & getters
│     ├── feature_pipeline.py
│     ├── extract_genres.py
│     └── hdf5_getters.py
│
├── MillionSongSubset_1000/     # Small (upload-safe) test subset
│
├── train_triplets_small.txt    # Small triplets subset (5000 rows)
│
├── notebooks/
│     └── analysis.ipynb        # Visualization & cluster/TDA analysis
│
├── out/                        # Generated: acoustic.parquet, umap, etc.
│
└── README.md


⸻

## 1. Quick Start (Using the Small 1000-Song Subset)

Create a virtual environment:

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

Run the small test pipeline:

export MSD_H5_DIR="./MillionSongSubset_1000"
export TASTE_TRIPLETS="./train_triplets_small.txt"
export SAMPLE_SONGS=1000

python msd_py3/feature_pipeline.py

This generates:

out/acoustic.parquet
out/acoustic_umap.parquet
out/behavior_edges.parquet
out/tda_summary.json

Everything should run in under 1 minute.

⸻

## 2. Full Dataset Reproduction Instructions (for instructors)

Step 1 — Download the full dataset

Option A: Prebuilt HDF5 subset (recommended)
Download from the official mirrors:
http://millionsongdataset.com/pages/getting-dataset/

You only need:

The HDF5 Song Files (7GB)
Taste Profile Subset (3GB)

Option B: AWS S3 (fastest)

aws s3 sync s3://millionsongdataset/data /your/local/path


⸻

Step 2 — Set environment variables

export MSD_H5_DIR="/path/to/MillionSongSubset"
export TASTE_TRIPLETS="/path/to/train_triplets.txt"

# Number of songs to use:
export SAMPLE_SONGS=10000     # default
export SAMPLE_SONGS=50000     # full experiment


⸻

Step 3 — Run the full pipeline

python msd_py3/feature_pipeline.py

Produces:
	•	acoustic.parquet — standardized acoustic features
	•	behavior_edges.parquet — co-listening graph
	•	acoustic_umap.parquet — 2D acoustic embedding + HDBSCAN clusters
	•	tda_summary.json — bottleneck distances (H₀, H₁ persistence)

⸻

## 3. Scaling Experiments

Change only this:

export SAMPLE_SONGS=50000

Everything else stays identical.

The code automatically:
	•	randomly samples files
	•	builds tables
	•	computes UMAP
	•	performs clustering
	•	computes TDA on a smaller subsample (TDA_SUBSAMPLE) for speed

⸻

## 4. Generating the 1000-Song Subset (If Needed)

Included for transparency.
This produced the MillionSongSubset_1000 folder in the repository:

import os, random, shutil

SRC = "./MillionSongSubset"
DST = "./MillionSongSubset_1000"
N = 1000

all_files = []
for root, _, files in os.walk(SRC):
    for f in files:
        if f.endswith(".h5"):
            all_files.append(os.path.join(root, f))

sample = random.sample(all_files, N)
for f in sample:
    rel = os.path.relpath(f, SRC)
    out = os.path.join(DST, rel)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    shutil.copy(f, out)

print("Done.")


⸻

## 5. Mini Triplets File

This file is included as train_triplets_small.txt.
It was generated using:

head -n 5000 train_triplets.txt > train_triplets_small.txt


⸻

## 6. Analysis Notebook (Clusters + TDA)

Open:

jupyter notebook notebooks/analysis.ipynb

This notebook includes:
	•	UMAP plots
	•	Cluster visualizations
	•	Distinctive genre analysis
	•	Representative artists
	•	Persistent homology diagrams (acoustic vs behavioral)

⸻

## 7. Citation

If using this in a paper or coursework:

Bertin-Mahieux, T. et al. (2011). The Million Song Dataset.

