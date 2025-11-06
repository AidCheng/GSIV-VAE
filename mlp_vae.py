#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
End-to-end training: MLP(7->3) + IV-VAE (3ch latent) + MLP(3->7)

支持：
- 冻结或联合训练 IV-VAE
- 自动尺寸对齐（例如 45x45 -> 40x40）
- 可梯度传播 (unwrapped encode/decode)
"""

import argparse, math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

# ✅ 直接使用你环境中的 IV-VAE 模块
from models.vae3d import IV_VAE


# =======================
# 工具函数
# =======================
def log(s): print(s, flush=True)
def set_seed(seed=42):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# =======================
# 数据与归一化
# =======================
def dict_to_features7(d: dict, device, dtype):
    xyz  = d["xyz"].to(device=device, dtype=dtype)
    fdc  = d["feature_dc_index"].to(device=device, dtype=dtype)
    chol = d["quant_cholesky_elements"].to(device=device, dtype=dtype)
    return torch.cat([xyz, fdc, chol], dim=-1)  # [N,7]


def build_dataset7(paths, device, dtype):
    feats = []
    for p in paths:
        d = torch.load(p, map_location="cpu")
        feats.append(dict_to_features7(d, device, dtype))
    X = torch.cat(feats, dim=0)
    x_min = X.amin(dim=0, keepdim=True)
    x_max = X.amax(dim=0, keepdim=True)
    X = (X - x_min) / (x_max - x_min + 1e-8)
    X = X * 2 - 1
    return X, x_min, x_max


# =======================
# 模型：MLP 编解码
# =======================
class GaussianEncoder(nn.Module):  # 7 -> 3
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 3)
        )
    def forward(self, x): return self.net(x)


class GaussianDecoder(nn.Module):  # 3 -> 7
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 7)
        )
    def forward(self, z): return self.net(z)


# =======================
# VAE 打包 & 解包
# =======================
def pack_latent3(z3, device):
    N = z3.shape[0]
    side = int(math.ceil(math.sqrt(N)))
    pad = side * side - N
    if pad > 0:
        z3 = torch.cat([z3, torch.zeros((pad, 3), device=z3.device)], dim=0)
    img = z3.view(side, side, 3).permute(2, 0, 1).unsqueeze(0).unsqueeze(2)
    return img.contiguous().to(device), side


def unpack_latent3(img):
    _, C, _, H, W = img.shape
    return img.squeeze(2).squeeze(0).permute(1, 2, 0).reshape(H * W, C)


# =======================
# 联合训练循环
# =======================
def train_joint(X, vae, enc, dec,
                epochs=200, batch_size=512, lr=5e-4, wd=1e-4,
                freeze_vae=False):

    device = next(enc.parameters()).device
    vae.train()
    for p in vae.parameters(): p.requires_grad_(not freeze_vae)

    opt = torch.optim.AdamW(
        list(enc.parameters()) + list(dec.parameters()) +
        ([] if freeze_vae else list(vae.parameters())),
        lr=lr, weight_decay=wd
    )
    loss_fn = nn.MSELoss()

    N = X.shape[0]
    for ep in range(1, epochs + 1):
        idx = torch.randperm(N, device=device)
        total_loss, total_count = 0, 0

        for i in range(0, N, batch_size):
            xb = X[idx[i:i+batch_size]]  # [B,7]
            z3 = enc(xb)                 # [B,3]

            # --- pack 到 VAE ---
            x_img, side = pack_latent3(z3, device)

            z_lat = vae.encode(x_img)

            # 如果 encode 返回 (mu, logvar)
            if isinstance(z_lat, (tuple, list)):
                mu, logvar = z_lat
                std = (0.5 * logvar).exp()
                eps = torch.randn_like(std)
                z_lat = mu + eps * std

            # ✅ 保留 [1, 8, 1, 2, 2]
            recon_img = vae.decode(z_lat)

            # --- unpack ---
            flat = unpack_latent3(recon_img)  # [H'*W',3]
            usable = min(flat.shape[0], xb.shape[0])
            flat = flat[:usable]
            xb = xb[:usable]

            recon7 = dec(flat)  # [usable,7]
            loss = loss_fn(recon7, xb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(enc.parameters()) + list(dec.parameters()) +
                ([] if freeze_vae else list(vae.parameters())), 1.0)
            opt.step()

            total_loss += loss.item() * usable
            total_count += usable

        avg = total_loss / total_count
        tag = "frozen-VAE" if freeze_vae else "joint"
        if ep % max(1, epochs // 10) == 0 or ep == 1 or ep == epochs:
            log(f"[{tag}] epoch {ep:4d}/{epochs} | loss={avg:.6f}")


# =======================
# 主入口
# =======================
def main():
    ap = argparse.ArgumentParser("Joint training MLP + IV-VAE + MLP")
    ap.add_argument("--data", nargs="+", required=True, help="comp_*.pth paths")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--freeze_vae", action="store_true")
    ap.add_argument("--save", type=str, default="joint_final_ckpt.pth")
    ap.add_argument("--zdim", type=int, default=8)
    ap.add_argument("--dim", type=int, default=64)
    args = ap.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # === Data ===
    X, _, _ = build_dataset7([Path(p) for p in args.data], device, dtype)
    log(f"[Data] {X.shape}")

    # === Models ===
    enc = GaussianEncoder(hidden=args.hidden).to(device)
    dec = GaussianDecoder(hidden=args.hidden).to(device)
    vae = IV_VAE(args.zdim, args.dim).to(device=device, dtype=dtype)
    log("[Init] IV-VAE + MLP ready.")

    # === Train ===
    train_joint(
        X, vae, enc, dec,
        epochs=args.epochs, batch_size=args.batch,
        lr=args.lr, wd=args.wd, freeze_vae=args.freeze_vae
    )

    torch.save({
        "encoder": enc.state_dict(),
        "decoder": dec.state_dict(),
        "vae": vae.state_dict() if not args.freeze_vae else None,
        "args": vars(args)
    }, args.save)
    log(f"[Save] Model saved to {args.save}")


if __name__ == "__main__":
    main()