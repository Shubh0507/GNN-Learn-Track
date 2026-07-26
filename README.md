# BrainSync

A Graph Attention Network (GAT) pipeline for EEG-based anxiety classification, built from scratch on top of PyTorch Geometric's `MessagePassing` class and evaluated on the DASPS dataset.

## Overview

BrainSync treats each EEG recording as a graph: electrodes are nodes, band-power features are node attributes, and edges are defined either by scalp proximity or by functional connectivity between channels. A custom multi-head GAT stack learns to classify each trial as anxious or non-anxious.

The project started as a from-scratch GNN curriculum (adjacency operations -> GCN normalization -> GAT -> pooling -> over-smoothing/WL test/GIN -> cross-validation strategy) and converged into a single working pipeline, `main_pipeline.py`, which is currently the source of truth. Earlier attempts to split the code across notebooks led to inconsistent state between cells, so everything now lives in one script.

## Architecture

- **`GCNLayer`** — a minimal from-scratch GCN layer (degree normalization + linear transform), kept as a reference/sanity-check implementation.
- **`GATLayer_SingleHeaded`** — single-head graph attention layer built directly on `torch_geometric.nn.MessagePassing`, implementing attention scoring and softmax normalization manually.
- **`GATLayer`** — multi-head version; each head has its own linear projection and attention vector, with outputs concatenated.
- **`BrainSync`** — a single-GAT-layer classifier (early prototype).
- **`BrainSync_8`** — the current model: 8 stacked GAT layers with residual connections back to a projected input (`prev_proj`) at every layer, followed by global mean pooling and a linear classification head. Residuals are used to counteract over-smoothing across depth.

## Data pipeline

1. **Synthetic DEAP-style data** — a 40-trial, 32-channel synthetic dataset used to validate the pipeline end-to-end before touching real data (band-power extraction, `Data` object creation, training loop).
2. **DASPS dataset** — the real target dataset (23 participants × 12 trials, 14 EEG channels each).
   - Raw signals are loaded from `.mat` files via `mat73`.
   - Anxiety labels are derived from HAM-A scores (HAM ≥ 25 → anxious).
   - **Band power features**: Welch PSD per channel, summed into delta/theta/alpha/beta/gamma bands, then z-normalized per subject to remove baseline/individual differences.
   - **Graph structure**: two approaches were implemented :-
     - *Proximity connectivity*: edges based on physical distance between electrode positions (10-20 montage), now deprecated in favor of the below.
     - *Functional connectivity* (current default): edges derived from band-averaged Pearson correlation between channels, thresholded relative to each node's row-mean correlation, with self-loops added so attention always includes each node's own features.

## Evaluation

Two evaluation strategies are implemented, both participant-grouped to avoid leakage (all trials from a given participant stay in the same split):

- **LOSO (Leave-One-Subject-Out)**: 23 folds, one held-out participant per fold, pooled predictions used for final precision/recall/F1.
- **Stratified 5-fold CV**: participants are split into 5 folds stratified by anxious/non-anxious label, run across 5 random seeds for stability, with a fresh model instantiated and class-weighted `CrossEntropyLoss` computed per fold to handle class imbalance.

## Results (as of July 26th)

| Setup | LOSO Accuracy | LOSO F1 | 5-Fold Accuracy | 5-Fold F1 |
|---|---|---|---|---|
| Before normalization | 0.453 ± 0.226 | 0.524 | 0.597 ± 0.092 | 0.657 ± 0.062 |
| After normalization | 0.504 ± 0.231 | 0.584 | 0.485 ± 0.035 | 0.517 ± 0.038 |
| After z-normalization | 0.496 ± 0.240 | 0.572 | 0.568 ± 0.010 | 0.630 ± 0.012 |
| **Functional connectivity (current)** | **0.576 ± 0.223** | **0.642** | **0.558 ± 0.039** | **0.617 ± 0.040** |

Functional-connectivity-based graphs currently give the best LOSO performance, suggesting that correlation-derived edges carry more discriminative signal for anxiety classification than fixed scalp-proximity edges.

## Repository structure

```
brainsync/
├── main_pipeline.py      # full pipeline: data loading, feature extraction,
│                         # graph construction, model, training, evaluation
├── dasps_data/           # DASPS raw + preprocessed .mat files (not tracked)
├── requrirements.txt     # Contains all relevant libraries
└── README.md
```

> Note: the pipeline currently lives as a single script/notebook-export (`main_pipeline.py`) rather than split modules, since splitting across notebooks previously caused state inconsistencies between cells during development. This also contains all the experimental stuff that was performed during the building of the pipeline(eg. containing the GCN pipeline as well)

## Requirements

- `torch`, `torch_geometric`
- `numpy`, `pandas`, `scipy`
- `mne` (for 10-20 montage electrode positions)
- `mat73` (for loading DASPS `.mat` files)
- `matplotlib` (for loss curve diagnostics)

## Usage

```bash
python main_pipeline.py
```

Running the script end-to-end will:
1. Load and preprocess the DASPS dataset.
2. Build functional-connectivity graphs per trial.
3. Run LOSO evaluation across all 23 participants.
4. Run stratified 5-fold cross-validation across 5 seeds.
5. Print accuracy and F1 summaries for both strategies.

<!-- ## Status / next steps

- Functional connectivity graphs outperform proximity-based graphs and are the current default.
- Class-weighted loss and participant-grouped splits are in place to control for imbalance and leakage.
- Potential next steps: hyperparameter sweep on GAT depth/heads, ablation on connectivity threshold, and comparison against a non-graph baseline (e.g. plain MLP on band powers) to quantify the benefit of the graph structure itself. -->