# train_ae_all_slides_10x.py
import os, json, numpy as np, h5py, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix, csc_matrix


SLIDES = {
    "17Q1": "/data2/sugandha/xenium_prostate/output-XETG00122__0010784__S18-9906-B27Us1_17Q1__20230912__220421/",
    "2Q1":  "/data2/sugandha/xenium_prostate/output-XETG00122__0010784__S18-9906-B27Us1_2Q1__20230912__220421/",
    "24Q1": "/data2/sugandha/xenium_prostate/output-XETG00122__0010787__S18-9906-B27Us1_24Q1__20230912__220421/",
    "9Q1":  "/data2/sugandha/xenium_prostate/output-XETG00122__0010787__S18-9906-B27Us1_9Q1__20230912__220421/",
}


TOPK_HVG       = 2000      # use top-K highly variable genes (if G<2000, uses all)
LATENT_DIM     = 32        # AE embedding size
BATCH_SIZE     = 512
EPOCHS_PER_SLIDE = 10
LR            = 1e-3


OUT_ROOT = list(SLIDES.values())[0]
AE_OUT   = os.path.join(OUT_ROOT, "ae_out_all"); os.makedirs(AE_OUT, exist_ok=True)


from scipy.sparse import csr_matrix  # if missing: pip install scipy

def load_10x_cells_by_genes(h5_path):
    """
    Returns:
      genes: list[str] (UPPERCASE, length G)
      X: np.ndarray float32 of shape (N_cells, G)  # dense
    Detects CSR vs CSC from indptr length and converts to (cells, genes).
    """
    with h5py.File(h5_path, "r") as f:
        mg = f["/matrix"]

        # genes (G)
        genes = mg["features"]["name"][()]
        genes = [g.decode() if isinstance(g,(bytes,bytearray)) else str(g) for g in genes]
        genes = [g.upper() for g in genes]

        # sparse pieces
        data    = mg["data"][()]
        indices = mg["indices"][()]
        indptr  = mg["indptr"][()]
        shape   = tuple(mg["shape"][()])  # (G, N)

        G, N = shape
        # Detect layout: CSR → len(indptr)=G+1; CSC → len(indptr)=N+1
        if len(indptr) == G + 1:
            # CSR (genes-major) -> transpose to (N,G)
            M = csr_matrix((data, indices, indptr), shape=shape, dtype=np.float32).T
        elif len(indptr) == N + 1:
            # CSC (cells-major in columns) -> transpose to (N,G)
            M = csc_matrix((data, indices, indptr), shape=shape, dtype=np.float32).T
        else:
            raise ValueError(f"indptr length {len(indptr)} not matching CSR/CSC for shape {shape}")

        X = M.toarray().astype(np.float32, copy=False)  # (N, G)
    return genes, X

def cpm_log1p(X):
    lib = np.clip(X.sum(1, keepdims=True), 1.0, None)
    return np.log1p(X / lib * 1e4)

# ------------------------------- Step A: pick Top-K HVGs -------------------------------
# Compute per-gene variance on CPM-log1p per slide, average across slides, take top-K
var_sum, var_cnt = {}, {}
all_gene_lists = []

for sid, root in SLIDES.items():
    h5 = os.path.join(root, "cell_feature_matrix.h5")
    genes, X = load_10x_cells_by_genes(h5)   # (N, G)
    all_gene_lists.append(genes)
    Xn = cpm_log1p(X)
    v = Xn.var(axis=0)                             # (G,)
    for g, vg in zip(genes, v):
        var_sum[g] = var_sum.get(g, 0.0) + float(vg)
        var_cnt[g] = var_cnt.get(g, 0) + 1

genes_all = list(var_sum.keys())
mean_var = np.array([var_sum[g] / var_cnt[g] for g in genes_all], dtype=np.float32)

k = min(TOPK_HVG, len(genes_all))                 # if 541 genes < 2000, you’ll keep all 541
top_idx = np.argsort(-mean_var)[:k]
GENES = sorted([genes_all[i] for i in top_idx])   # sorted for stable ordering

json.dump(GENES, open(os.path.join(AE_OUT, "ae_genes.json"), "w"))
print(f"[HVG] selected {len(GENES)} genes -> {os.path.join(AE_OUT,'ae_genes.json')}")

# ------------------------------- AE model -------------------------------
class AE(nn.Module):
    def __init__(self, d, k=32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(d, 1024), nn.ReLU(),
            nn.Linear(1024, 256), nn.ReLU(),
            nn.Linear(256, k)
        )
        self.dec = nn.Sequential(
            nn.Linear(k, 256), nn.ReLU(),
            nn.Linear(256, 1024), nn.ReLU(),
            nn.Linear(1024, d)
        )
    def encode(self, x): return self.enc(x)
    def forward(self, x): z = self.enc(x); y = self.dec(z); return y, z

device = "cuda" if torch.cuda.is_available() else "cpu"
model  = AE(d=len(GENES), k=LATENT_DIM).to(device)
opt    = torch.optim.Adam(model.parameters(), lr=LR)
lossfn = nn.MSELoss()

class Cells(Dataset):
    def __init__(self, X): self.X = X
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i):
        x = self.X[i]
        return torch.from_numpy(x), torch.from_numpy(x)

def align_cols(genes_src, X_src, genes_target):
    idx = {g:i for i,g in enumerate(genes_src)}
    out = np.zeros((X_src.shape[0], len(genes_target)), dtype=np.float32)
    take = [idx[g] for g in genes_target if g in idx]
    put  = [i       for i,g in enumerate(genes_target) if g in idx]
    if take:
        out[:, put] = X_src[:, take]
    return out

def zscore_cols(X):
    mu = X.mean(0, keepdims=True)
    sigma = X.std(0, keepdims=True) + 1e-6
    return (X - mu) / sigma, mu, sigma

# ------------------------------- Step B: train sequentially over slides -------------------------------
all_mus, all_sigmas = [], []

for sid, root in SLIDES.items():
    h5 = os.path.join(root, "cell_feature_matrix.h5")
    genes_s, X_s = load_10x_cells_by_genes(h5)  # (N, Gs)
    # align to global HVG list
    A = align_cols(genes_s, X_s, GENES)             # (N, |GENES|)
    Xn = cpm_log1p(A)                               
    Xz, mu, sigma = zscore_cols(Xn)                 # per-slide z-score (we’ll save average μ/σ)
    all_mus.append(mu); all_sigmas.append(sigma)

    dl = DataLoader(Cells(Xz), batch_size=BATCH_SIZE, shuffle=True,
                    num_workers=2, pin_memory=True)
    for ep in range(1, EPOCHS_PER_SLIDE+1):
        model.train(); tot = 0.0
        for xb, yb in dl:
            xb = xb.to(device); yb = yb.to(device)
            yhat, _ = model(xb)
            loss = lossfn(yhat, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * xb.size(0)
        print(f"{sid}  epoch {ep}/{EPOCHS_PER_SLIDE}  recon_mse={tot/len(dl.dataset):.6f}")

# ------------------------------- Save artifacts -------------------------------
mu    = np.mean(np.stack(all_mus,   axis=0), axis=0)
sigma = np.mean(np.stack(all_sigmas,axis=0), axis=0)
np.savez(os.path.join(AE_OUT, "scaler.npz"), mu=mu, sigma=sigma)

torch.save(model.state_dict(), os.path.join(AE_OUT, "ae.ckpt"))

print("Saved:")
print("  ", os.path.join(AE_OUT, "ae.ckpt"))
print("  ", os.path.join(AE_OUT, "ae_genes.json"))
print("  ", os.path.join(AE_OUT, "scaler.npz"))
