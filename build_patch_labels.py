import os, glob, json, numpy as np, pandas as pd, h5py
from scipy.sparse import csr_matrix, csc_matrix


HE_OUT_DIR = "/data2/sugandha/he_prostate_patches" 
IMG_DIR    = os.path.join(HE_OUT_DIR, "images")

SLIDES = {
    "17Q1": "/data2/sugandha/xenium_prostate/output-XETG00122__0010784__S18-9906-B27Us1_17Q1__20230912__220421/",
    "2Q1":  "/data2/sugandha/xenium_prostate/output-XETG00122__0010784__S18-9906-B27Us1_2Q1__20230912__220421/",
    "24Q1": "/data2/sugandha/xenium_prostate/output-XETG00122__0010787__S18-9906-B27Us1_24Q1__20230912__220421/",
    "9Q1":  "/data2/sugandha/xenium_prostate/output-XETG00122__0010787__S18-9906-B27Us1_9Q1__20230912__220421/",
}

OUT_CSV = os.path.join(HE_OUT_DIR, "he_patches_labels_fusion_all_fixed_6Nov.csv")

# ---------- KNOBS ----------
PATCH_SIZE_L0       = 512
MIN_CELLS           = 10     # drop sparse patches
MIN_GENES_PER_SIG   = 3      # per-cell: need at least this many genes from the panel

# ---------- SIGNATURES (same 6) ----------
SIGNATURES = {
  "Immune_T": ["CD3D","CD3E","CD2","CD8A","TRAC","GZMB","GZMA","PRF1","NKG7","KLRD1","CXCL9","CXCL10","HLA-DRA","HLA-DPA1","PTPRC"],
  "Immune_B": ["MS4A1","CD79A","CD79B","MZB1","BANK1"],
  "Pericyte": ["RGS5","PDGFRB","ACTA2","TAGLN","MYH11","CNN1"],
  "SmoothMuscle": ["ACTA2","TAGLN","MYH11","CNN1","MYLK","DMD"],
  "Fibroblast": ["FAP","PDGFRA","PDGFRB","COL3A1","DCN","LUM","THY1","CXCL12"],
  "Proliferation": ["MKI67","TOP2A","PCNA","TK1","TYMS","UBE2C","CCNA2","CCNB1","CDC20","AURKA","BUB1"],
}
SIGNATURES = {k:[g.upper() for g in v] for k,v in SIGNATURES.items()}
SIGS = list(SIGNATURES.keys())


def load_all_manifests(he_dir: str):
    """
    Combine all manifest CSVs in HE_OUT_DIR.
    Supports having one manifest per slide. Keeps only 512x512 windows.
    """
    csvs = sorted(glob.glob(os.path.join(he_dir, "patches_manifest*.csv")))
    if not csvs:
        raise FileNotFoundError("No patches_manifest*.csv found in HE_OUT_DIR.")
    frames = []
    for p in csvs:
        m = pd.read_csv(p)
        if not {"y0","x0","h","w"}.issubset(m.columns):
            raise ValueError(f"Manifest missing columns: {p}")
        m = m[(m["h"] == PATCH_SIZE_L0) & (m["w"] == PATCH_SIZE_L0)].copy()
        m["key"] = m["y0"].astype(int).astype(str) + ":" + m["x0"].astype(int).astype(str)
        frames.append(m[["y0","x0","key"]])
    out = pd.concat(frames, ignore_index=True).drop_duplicates()
    return set(out["key"].tolist())

def load_cells_parquet(slide_root):
    p = os.path.join(slide_root, "cells.parquet")
    df = pd.read_parquet(p)[["cell_id", "x_centroid", "y_centroid"]].copy()
    df.rename(columns={"x_centroid":"x", "y_centroid":"y"}, inplace=True)
    df["x"] = df["x"].astype("float32"); df["y"] = df["y"].astype("float32")
    return df  # (cell_id, x, y)

def load_10x_cells_by_genes_with_ids(h5_path):
    """
    Returns:
      genes:   [G] str (UPPERCASE)
      cellids: [N] str  (barcodes)
      X:       (N, G) float32 dense counts
    Detects CSR vs CSC via indptr length and transposes to (cells, genes).
    """
    with h5py.File(h5_path, "r") as f:
        mg = f["/matrix"]

        genes = mg["features"]["name"][()]
        genes = [g.decode() if isinstance(g,(bytes,bytearray)) else str(g) for g in genes]
        genes = [g.upper() for g in genes]

        barcodes = mg["barcodes"][()]
        cellids  = [b.decode() if isinstance(b,(bytes,bytearray)) else str(b) for b in barcodes]

        data, indices, indptr = mg["data"][()], mg["indices"][()], mg["indptr"][()]
        G, N = tuple(mg["shape"][()])  # (genes, cells)

        if len(indptr) == G + 1:      # CSR (genes-major)
            M = csr_matrix((data, indices, indptr), shape=(G, N), dtype=np.float32).T
        elif len(indptr) == N + 1:    # CSC (cells-major)
            M = csc_matrix((data, indices, indptr), shape=(G, N), dtype=np.float32).T
        else:
            raise ValueError("indptr length does not match CSR/CSC")
        X = M.toarray().astype(np.float32, copy=False)  # (N, G)
    return genes, cellids, X

def cpm_log1p(X):
    lib = np.clip(X.sum(1, keepdims=True), 1.0, None)
    return np.log1p(X / lib * 1e4)

def zscore_cols(X):
    mu = X.mean(0, keepdims=True)
    sigma = X.std(0, keepdims=True) + 1e-6
    return (X - mu) / sigma

def assign_patches(cells_df):
    px = (cells_df["x"] // PATCH_SIZE_L0).astype("int32")
    py = (cells_df["y"] // PATCH_SIZE_L0).astype("int32")
    cells_df["x0"] = px * PATCH_SIZE_L0
    cells_df["y0"] = py * PATCH_SIZE_L0
    cells_df["key"] = cells_df["y0"].astype(str) + ":" + cells_df["x0"].astype(str)
    return cells_df


def find_png(y0, x0, sid):
    hits = glob.glob(os.path.join(IMG_DIR, f"{sid}_y{int(y0)}_x{int(x0)}.png"))
    return hits[0] if hits else ""
    
def find_ae(y0, x0, sid):
    ae_dir = os.path.join(HE_OUT_DIR, "ae_npys_new")
    pattern = os.path.join(ae_dir, f"{sid}_y{int(y0)}_x{int(x0)}.npy")
    hits = glob.glob(pattern)
    return hits[0] if hits else ""
# ---------- LOAD KEPT PATCHES (all manifests) ----------
kept_keys = load_all_manifests(HE_OUT_DIR)
print("Total kept keys across manifests:", len(kept_keys))


rows = []
per_slide_counts = {}

for sid, root in SLIDES.items():
    print(f"\n=== {sid} ===")

   
    cells = load_cells_parquet(root)  # (cell_id, x, y)
    print("cells.parquet rows:", len(cells))

    
    genes, h5_cellids, X = load_10x_cells_by_genes_with_ids(os.path.join(root, "cell_feature_matrix.h5"))
    print("H5: cells x genes:", X.shape)

   
    Xz = zscore_cols(cpm_log1p(X))


    expr = pd.DataFrame(Xz, columns=genes)
    expr.insert(0, "cell_id", h5_cellids)
    df = cells.merge(expr, on="cell_id", how="inner")
    print("merged rows:", len(df))
    if len(df) == 0:
        print("NO overlap of cell_ids between cells.parquet and H5 — skipping slide.")
        continue

   
    df = assign_patches(df)
    df = df[df["key"].isin(kept_keys)].reset_index(drop=True)
    print("cells inside kept patches:", len(df))
    if len(df) == 0:
        continue

    # 6) per-cell signature scores (need ≥ MIN_GENES_PER_SIG)
    for name, panel in SIGNATURES.items():
        present = [g for g in panel if g in df.columns]
        if not present:
            df[name] = np.nan
            continue
        avail_ct = df[present].notna().sum(axis=1)
        df[name] = df[present].mean(axis=1).where(avail_ct >= MIN_GENES_PER_SIG, np.nan)

    # 7) aggregate to patch (NaN-robust median), require MIN_CELLS + PNG + AE
    kept_rows = 0
    for (y0, x0), g in df.groupby(["y0","x0"], sort=False):
        if len(g) < MIN_CELLS:
            continue
        png_path = find_png(y0, x0,sid)
        ae_path  = find_ae(y0, x0,sid)
        if not png_path or not ae_path:
            continue
        sig_meds = {s: float(np.nanmedian(g[s].to_numpy())) for s in SIGS}
        rows.append({
            "slide": sid,
            "y0": int(y0), "x0": int(x0),
            "size_l0": PATCH_SIZE_L0,
            "n_cells": int(len(g)),
            "png_path": png_path,
            "ae_path":  ae_path,
            **sig_meds
        })
        kept_rows += 1

    per_slide_counts[sid] = kept_rows
    print(f"patch rows kept for {sid}:", kept_rows)

labels = pd.DataFrame(rows)
labels.to_csv(OUT_CSV, index=False)

print("\nWROTE:", OUT_CSV, "rows=", len(labels))
print("Per-signature non-NaN %:", (~labels[SIGS].isna()).mean().mul(100).round(1).to_dict())
print("Rows per slide:", per_slide_counts)
