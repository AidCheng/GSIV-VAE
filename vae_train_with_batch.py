import torch
import torch.nn.functional as F
import math
from pathlib import Path
import argparse
from tqdm import tqdm
from models.guassian_iv_vae import IV_VAE


# ======================
# 数据打包与解包函数
# ======================
def pack_7d_to_image(X7: torch.Tensor, tile: int = 2):
    """
    把 [N,7] 数据打包为 [1,3,1,H,W] 图像。
    """
    device, dtype = X7.device, X7.dtype
    N = X7.shape[0]
    C = 3
    cap = tile * tile * C
    assert cap >= 7, f"tile={tile} capacity too low (cap={cap})！"

    G = int(math.ceil(math.sqrt(N)))
    pad_n = G * G - N
    if pad_n > 0:
        X7 = torch.cat([X7, torch.zeros(pad_n, 7, device=device, dtype=dtype)], dim=0)

    Xcap = torch.zeros((X7.size(0), cap), device=device, dtype=dtype)
    Xcap[:, :7] = X7

    H = W = G * tile
    img = torch.zeros((1, C, 1, H, W), device=device, dtype=dtype)
    idx = 0
    for gy in range(G):
        for gx in range(G):
            base_y, base_x = gy * tile, gx * tile
            vec = Xcap[idx]; k = 0
            for py in range(tile):
                for px in range(tile):
                    for ch in range(C):
                        if k < cap:
                            img[0, ch, 0, base_y + py, base_x + px] = vec[k]
                            k += 1
            idx += 1
    return img, {"G": G, "tile": tile, "N": N, "cap": cap}


def unpack_image_to_7d(img: torch.Tensor, meta: dict):
    """
    将 [1,3,1,H,W] 解包回 [N,7]
    """
    device, dtype = img.device, img.dtype
    G, tile, N, cap = meta["G"], meta["tile"], meta["N"], meta["cap"]
    C = 3
    H, W = img.shape[-2:]
    Xcap = torch.zeros((G * G, cap), device=device, dtype=dtype)
    idx = 0
    for gy in range(G):
        for gx in range(G):
            base_y, base_x = gy * tile, gx * tile
            k = 0
            for py in range(tile):
                for px in range(tile):
                    if base_y + py >= H or base_x + px >= W:
                        continue
                    for ch in range(C):
                        if k >= cap:
                            break
                        Xcap[idx, k] = img[0, ch, 0, base_y + py, base_x + px]
                        k += 1
            idx += 1
    X7 = Xcap[:, :7]
    return X7[:N]


# ======================
# 批次化训练
# ======================
def train_and_eval_batch(vae, X, device, tile=2, epochs=500, lr=1e-4,
                         batch_size=256, save_path="vae_tile_trained.pth"):

    X7 = 2 * (X - X.min()) / (X.max() - X.min() + 1e-8) - 1
    vae = vae.to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(vae.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    N = X7.shape[0]

    best_mse = float("inf")
    for epoch in range(1, epochs + 1):
        vae.train()
        perm = torch.randperm(N)
        total_loss = 0.0

        for i in tqdm(range(0, N, batch_size), desc=f"[Epoch {epoch}]"):
            batch_idx = perm[i:i + batch_size]
            xb = X7[batch_idx].to(device)
            x_img, meta = pack_7d_to_image(xb, tile)
            z = vae.encode(x_img)

            if isinstance(z, (tuple, list)):
                mu, logvar = z
                std = (0.5 * logvar).exp()
                eps = torch.randn_like(std)
                z = mu + eps * std

            recon = vae.decode(z)
            X7_recon = unpack_image_to_7d(recon, meta)
            loss = loss_fn(X7_recon, xb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / (N // batch_size + 1)
        if avg_loss < best_mse:
            best_mse = avg_loss
            torch.save(vae.state_dict(), f"epoch_{epoch}_best_{best_mse:.6f}.pth")

        if epoch % 10 == 0 or epoch == 1:
            print(f"[Train] Epoch {epoch}/{epochs} | loss={avg_loss:.6f}")

    torch.save(vae.state_dict(), save_path)
    print(f"[Save] model saved to {save_path}")


# ======================
# 批次化评估
# ======================
def eval_only_batch(vae, X, device, tile=2, batch_size=256,
                    save_recon_path="./results/gaussians/recon_output.pth"):
    X7 = 2 * (X - X.min()) / (X.max() - X.min() + 1e-8) - 1
    vae = vae.to(device=device, dtype=torch.float32)
    vae.eval()
    N = X7.shape[0]

    all_recon = []
    with torch.no_grad():
        for i in tqdm(range(0, N, batch_size), desc="[Eval]"):
            xb = X7[i:i + batch_size].to(device)
            x_img, meta = pack_7d_to_image(xb, tile)
            z = vae.encode(x_img)

            if isinstance(z, (tuple, list)):
                mu, logvar = z
                std = (0.5 * logvar).exp()
                eps = torch.randn_like(std)
                z = mu + eps * std

            recon = vae.decode(z)
            X7_recon = unpack_image_to_7d(recon, meta)
            all_recon.append(X7_recon.cpu())

    X7_recon_all = torch.cat(all_recon, dim=0)[:N]
    mse = F.mse_loss(X7_recon_all, X7.cpu())
    print(f"[Eval] MSE = {mse.item():.6f}")

    torch.save(X7_recon_all, save_recon_path)
    print(f"[Save] recon results -> {save_recon_path}")
    return X7_recon_all


# ======================
# 主程序
# ======================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="comp_0.pth")
    parser.add_argument("--tile", type=int, default=2)
    parser.add_argument("--mode", type=str, default="eval", choices=["eval", "train"])
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--zdim", type=int, default=64)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--ckpt", type=str, default="")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    comp = torch.load(Path(args.data))
    xyz = comp["xyz"].to(device, dtype=torch.float32)
    fdc = comp["feature_dc_index"].to(device, dtype=torch.float32)
    chol = comp["quant_cholesky_elements"].to(device, dtype=torch.float32)
    X7 = torch.cat([xyz, fdc, chol], dim=-1)

    print(f"[Data Size] {X7.shape}")

    vae = IV_VAE(args.zdim, args.dim)
    if args.ckpt and Path(args.ckpt).exists():
        vae.load_state_dict(torch.load(args.ckpt, map_location=device))
        print(f"[Init] Loaded ckpt: {args.ckpt}")

    if args.mode == "train":
        train_and_eval_batch(
            vae, X7, device,
            tile=args.tile, epochs=args.epochs, lr=args.lr,
            batch_size=args.batch_size)
    else:
        eval_only_batch(
            vae, X7, device,
            tile=args.tile, batch_size=args.batch_size)


if __name__ == "__main__":
    main()