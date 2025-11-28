# HE–Xenium gene-signature ViT (exploratory)

This repository contains early exploratory code for learning **molecular
signatures from H&E patches** by combining:

- an **autoencoder** trained on Xenium gene expression, and  
- a **ViT-style backbone** applied to co-registered H&E patches.

The goal is to regress patch-level **gene-signature scores** (e.g. immune,
stromal, proliferation) from histology while using Xenium-derived embeddings
as an auxiliary input. This is a prototype / proof-of-concept and **not** a
final research pipeline.

---

## High-level idea

1. **Autoencoder on Xenium gene expression**

   - Load 10x-style `cell_feature_matrix.h5` for multiple slides.
   - Select top highly variable genes across slides.
   - Train a fully connected autoencoder to obtain a **latent embedding** for
     each cell.
   - Save:
     - `ae.ckpt` — trained AE weights  
     - `ae_genes.json` — list of genes used  
     - `scaler.npz` — mean / std for z-scoring

2. **Cell → patch aggregation**

   - Use `cells.parquet` (with `x_centroid`, `y_centroid`) to map cells to
     fixed-size H&E patches (e.g. 512×512).
   - Encode each cell with the AE, then average cell embeddings within each
     patch → one **AE vector per patch**, saved as `.npy`.

3. **Patch-level labels from gene signatures**

   - Define a set of gene-signature panels (e.g. `Immune_T`, `Pericyte`,
     `SmoothMuscle`, etc.).
   - Normalize expression (CPM → log1p → z-score).
   - For each patch, aggregate per-cell signature scores to a summary value
     (e.g. median per signature).
   - Build a CSV with:
     - `png_path` — path to H&E patch image  
     - `ae_path` — path to AE embedding `.npy`  
     - columns for signature scores

4. **HE + AE ViT model**

   - A ViT-style backbone processes the H&E patch into a sequence of tokens.
   - The **class token** is fused with the AE embedding (via a small linear
     projection).
   - A regression head predicts multiple signature scores per patch.

---

## Scripts

- `train_ae_xenium.py`  
  Train an autoencoder on Xenium cell-by-gene matrices across slides, select
  highly variable genes, and save `ae.ckpt`, `ae_genes.json`, and `scaler.npz`.

- `encode_cells_to_patches.py`  
  Load trained AE, encode cells, aggregate to patch-level embeddings based on
  spatial coordinates from `cells.parquet`, and save one `.npy` per patch.

- `build_patch_labels.py`  
  Combine H&E patch manifests, Xenium expression, and predefined gene
  signatures to build a patch-level CSV with:
  `png_path`, `ae_path`, `n_cells`, and signature values.

- `train_he_ae_vit.py`  
  Define the HE+AE ViT model, create a dataset from the patch-level CSV, and
  train the model to regress selected gene-signature scores.

This code was run on a cluster environment; paths in the scripts are specific
to that setup and should be adjusted for your own data layout.

---

## Notes

This repository contains an **initial exploratory implementation** used for
prototyping and early experiments.

The ViT-style backbone is adapted from 
[vit-pytorch](https://github.com/lucidrains/vit-pytorch) implementation by
Phil Wang (lucidrains)
