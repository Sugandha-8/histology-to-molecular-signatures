import os, json, numpy as np, pandas as pd, torch, h5py
from scipy.sparse import csr_matrix, csc_matrix
from torch import nn

# --- paths ---
SLIDES = {
    "17Q1": "/data2/sugandha/xenium_prostate/output-XETG00122__0010784__S18-9906-B27Us1_17Q1__20230912__220421/",
    "2Q1":  "/data2/sugandha/xenium_prostate/output-XETG00122__0010784__S18-9906-B27Us1_2Q1__20230912__220421/",
    "24Q1": "/data2/sugandha/xenium_prostate/output-XETG00122__0010787__S18-9906-B27Us1_24Q1__20230912__220421/",
    "9Q1":  "/data2/sugandha/xenium_prostate/output-XETG00122__0010787__S18-9906-B27Us1_9Q1__20230912__220421/",
}

AE_DIR = list(SLIDES.values())[0] + "/ae_out_all"
AE_GENES = json.load(open(f"{AE_DIR}/ae_genes.json"))
scaler = np.load(f"{AE_DIR}/scaler.npz")
mu, sigma = scaler["mu"], scaler["sigma"]

PATCH_SIZE = 512
AE_NPY_DIR = "/data2/sugandha/he_prostate_patches/ae_npys_new"
os.makedirs(AE_NPY_DIR, exist_ok=True)

# --- AE definition (must match training) ---
class AE(nn.Module):
    def __init__(self, d, k=32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(d, 1024), nn.ReLU(),
            nn.Linear(1024, 256), nn.ReLU(),
            nn.Linear(256, k)
        )
    def encode(self, x): return self.enc(x)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
print(torch.cuda.get_device_name(0))
model = AE(len(AE_GENES), 32).to(device)
#model.load_state_dict(torch.load(f"{AE_DIR}/ae.ckpt", map_location=device))
#model.eval()
ckpt = torch.load(f"{AE_DIR}/ae.ckpt", map_location=device)
model.load_state_dict(ckpt, strict=False)
model.eval()
def load_10x(h5_path):
    with h5py.File(h5_path, "r") as f:
        mg = f["/matrix"]
        data, idx, indptr = mg["data"][()], mg["indices"][()], mg["indptr"][()]
        G, N = mg["shape"][()]
        if len(indptr) == G + 1:
            M = csr_matrix((data, idx, indptr), shape=(G, N)).T
        else:
            M = csc_matrix((data, idx, indptr), shape=(G, N)).T
        genes = [g.decode() for g in mg["features"]["name"][()]]
        barcodes = [b.decode() for b in mg["barcodes"][()]]
    return [g.upper() for g in genes], barcodes, M.toarray().astype(np.float32)

def align_cols(genes_src, X_src, genes_target):
    idx = {g:i for i,g in enumerate(genes_src)}
    out = np.zeros((X_src.shape[0], len(genes_target)), np.float32)
    take = [idx[g] for g in genes_target if g in idx]
    put  = [i for i,g in enumerate(genes_target) if g in idx]
    out[:, put] = X_src[:, take]
    return out

for sid, root in SLIDES.items():
    print(f"=== {sid} ===")
    genes, cellids, X = load_10x(os.path.join(root, "cell_feature_matrix.h5"))
    coords = pd.read_parquet(os.path.join(root, "cells.parquet"))[["cell_id","x_centroid","y_centroid"]]
    coords.rename(columns={"x_centroid":"x","y_centroid":"y"}, inplace=True)

    A = align_cols(genes, X, AE_GENES)
    Xn = np.log1p((A / np.clip(A.sum(1, keepdims=True), 1, None)) * 1e4)
    Xz = (Xn - mu) / sigma

    with torch.no_grad():
        Z_list = []
        bs = 1024  
        for i in range(0, Xz.shape[0], bs):
            xb = torch.from_numpy(Xz[i:i+bs]).to(device)
            zb = model.encode(xb).cpu().numpy()
            Z_list.append(zb)
            del xb, zb  # free GPU memory per loop
            torch.cuda.empty_cache()
        Z = np.concatenate(Z_list, axis=0)
        del Z_list

        

    coords["cell_id"] = cellids
    coords["x0"] = (coords["x"] // PATCH_SIZE).astype(int) * PATCH_SIZE
    coords["y0"] = (coords["y"] // PATCH_SIZE).astype(int) * PATCH_SIZE

    for (y0,x0), g in coords.groupby(["y0","x0"]):
        patch_vec = Z[g.index].mean(0)
        np.save(f"{AE_NPY_DIR}/{sid}_y{y0}_x{x0}.npy", patch_vec)

    print("saved AE npy per patch for", sid)
