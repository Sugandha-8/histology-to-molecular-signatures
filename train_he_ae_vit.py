import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
from PIL import Image


CSV_PATH   = "/data2/sugandha/he_prostate_patches/he_patches_labels_fusion_all_fixed_6Nov.csv"
IMG_SIZE   = 512
AE_DIM     = 32
VIT_DIM    = 128
OUT_DIM    = 3  # Immune_T, Pericyte, SmoothMuscle
BATCH_SIZE = 1
LR         = 2e-4
EPOCHS     = 10
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))       


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim), nn.Dropout(dropout)
        )
    def forward(self, x): return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner = dim_head * heads
        self.heads, self.scale = heads, dim_head ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, inner * 3, bias=False)
        self.attn = nn.Softmax(dim=-1)
        self.out = nn.Sequential(nn.Linear(inner, dim), nn.Dropout(dropout))
    def forward(self, x):
        x = self.norm(x)
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        B, N, C = q.shape
        q = q.view(B, N, self.heads, -1).transpose(1, 2)
        k = k.view(B, N, self.heads, -1).transpose(1, 2)
        v = v.view(B, N, self.heads, -1).transpose(1, 2)
        dots = (q @ k.transpose(-1, -2)) * self.scale
        a = self.attn(dots)
        out = (a @ v).transpose(1, 2).reshape(B, N, C)
        return self.out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Attention(dim, heads, dim_head, dropout),
                FeedForward(dim, mlp_dim, dropout)
            ]) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        for attn, ff in self.layers:
            x = x + attn(x)
            x = x + ff(x)
        return self.norm(x)
        
class HEViTBackbone(nn.Module):
    def __init__(self, image_size=512, patch_size=32, dim=256, depth=4,
                 heads=4, mlp_dim=512, dim_head=64, emb_dropout=0.):
        super().__init__()
        self.patch_size = patch_size 
        assert image_size % patch_size == 0
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size * patch_size

        # patch projection (no unfold inside)
        self.to_patch = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        self.cls = nn.Parameter(torch.randn(1, 1, dim))
        self.pos = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.drop = nn.Dropout(emb_dropout)
        self.tr = Transformer(dim, depth, heads, dim_head, mlp_dim, emb_dropout)
        self.dim = dim

    def forward_tokens(self, img):
        B = img.size(0)
        # 1. Create patches manually with unfold
        patches = nn.functional.unfold(img, kernel_size=self.patch_size, stride=self.patch_size)  # [B, 3*p*p, N]
        patches = patches.transpose(1, 2)                               # [B, N, 3*p*p]
        # 2. Project patches through linear layers
        patches = self.to_patch(patches)                                # [B, N, dim]
        N = patches.size(1)
        # 3. Add CLS token and positional encoding
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, patches], dim=1) + self.pos[:, :N + 1]
        x = self.drop(x)
        return self.tr(x)


    
class HE_AE_ViT(nn.Module):
    def __init__(self, ae_dim=32, vit_dim=128, **vit_kw):
        super().__init__()
        self.vit = HEViTBackbone(dim=vit_dim, **vit_kw)
        self.ae_proj = nn.Linear(ae_dim, vit_dim)
        self.head = nn.Linear(vit_dim, OUT_DIM)
    def forward(self, img, ae_vec):
        x = self.vit.forward_tokens(img)
        cls, tokens = x[:, :1, :], x[:, 1:, :]
        cls = cls + self.ae_proj(ae_vec).unsqueeze(1)
        x = torch.cat([cls, tokens], dim=1)
        y = self.head(x[:, 0, :])
        return y


tx = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.ConvertImageDtype(torch.float32)
])

class PatchDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[["png_path", "ae_path", "Immune_T", "Pericyte", "SmoothMuscle"]].dropna(subset=["ae_path"])
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = tx(Image.open(row.png_path).convert('RGB'))
        ae = torch.tensor(np.load(row.ae_path), dtype=torch.float32)
        y = torch.tensor([row.Immune_T, row.Pericyte, row.SmoothMuscle], dtype=torch.float32)
        return img, ae, y

def collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    aes  = torch.stack([b[1] for b in batch])
    y    = torch.stack([b[2] for b in batch])
    mask = ~torch.isnan(y)
    y[~mask] = 0.0
    return imgs, aes, y, mask


full_ds = PatchDataset(CSV_PATH)
n = len(full_ds)
n_train, n_val, n_test = int(0.8*n), int(0.1*n), n - int(0.8*n) - int(0.1*n)
train_ds, val_ds, test_ds = random_split(full_ds, [n_train, n_val, n_test])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)
print("len train",len(train_ds))
print(len(val_ds))
print(len(test_ds))

def masked_mse(pred, target, mask):
    diff = (pred - target)**2
    diff = diff[mask]
    return diff.mean() if diff.numel() > 0 else torch.tensor(0., device=pred.device)

model = HE_AE_ViT(ae_dim=AE_DIM, vit_dim=VIT_DIM, image_size=IMG_SIZE,
                  patch_size=32, depth=4, heads=4, mlp_dim=512).to(DEVICE)
print(model.vit.patch_size)  
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)

for epoch in range(EPOCHS):
    model.train(); total_loss = 0
    for imgs, aes, y, mask in train_loader:
        imgs, aes, y, mask = imgs.to(DEVICE), aes.to(DEVICE), y.to(DEVICE), mask.to(DEVICE)
        pred = model(imgs, aes)
        loss = masked_mse(pred, y, mask)
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}  TrainLoss={total_loss/len(train_loader):.4f}")
