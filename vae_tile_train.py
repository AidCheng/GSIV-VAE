import torch
import torch.nn.functional as F
import math
from pathlib import Path
import argparse
from models.guassian_iv_vae import Custom_IV_VAE as IV_VAE



def pack_7d_to_image(X7: torch.Tensor, tile: int = 2):
    """
    把 [N,7] 数据打包为 [1,3,1,H,W] 图像，用 micro-tile = tile×tile 嵌入。
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
    将 [1, 3, 1, H, W] 解包回 [N, 7]
    """
    device, dtype = img.device, img.dtype
    G, tile, N, cap = meta["G"], meta["tile"], meta["N"], meta["cap"]
    C = 3
    H, W = img.shape[-2:]  # 图像高宽

    Xcap = torch.zeros((G * G, cap), device=device, dtype=dtype)
    idx = 0

    for gy in range(G):
        for gx in range(G):
            base_y, base_x = gy * tile, gx * tile
            k = 0
            for py in range(tile):
                for px in range(tile):
                    y, x = base_y + py, base_x + px
                    if y >= H or x >= W:
                        # 超出边界的 tile 跳过
                        continue
                    for ch in range(C):
                        if k >= cap:
                            break
                        Xcap[idx, k] = img[0, ch, 0, y, x]
                        k += 1
            idx += 1

    X7 = Xcap[:, :7]
    return X7[:N]


def train_and_eval(vae, X, device, tile=2, epochs=500, lr=1e-4, save_path="vae_tile_trained.pth"):
    X7 = 2 * (X - X.min()) / (X.max() - X.min() + 1e-8) - 1
    vae = vae.to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(vae.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    x_img, meta = pack_7d_to_image(X7, tile)
    print(f"[Pack] -> {x_img.shape}")

    best_mse = 1
    for epoch in range(1, epochs + 1):
        vae.train()
        z = vae.encode(x_img)
        if isinstance(z, (tuple, list)):
            mu, logvar = z
            std = (0.5 * logvar).exp()
            eps = torch.randn_like(std)
            z = mu + eps * std
        recon = vae.decode(z)
        X7_recon = unpack_image_to_7d(recon, meta)
        X7_recon_denorm = (X7_recon + 1) / 2 * (X.max() - X.min()) + X.min()
        loss = loss_fn(X7_recon_denorm, X)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss.item() < best_mse:
            best_mse = loss.item()
            torch.save(vae.state_dict(), f"epoch_{epoch}_best_vae_mse{best_mse}.pth")

        if epoch % 20 == 0 or epoch == 1:
            print(f"[Train] epoch {epoch:4d}/{epochs} | loss={loss.item():.6f}")

    torch.save(vae.state_dict(), save_path)
    print(f"[Save] model saved to {save_path}")

    vae.eval()
    with torch.no_grad():
        z = vae.encode(x_img)
        if isinstance(z, (tuple, list)):
            mu, logvar = z
            std = (0.5 * logvar).exp()
            eps = torch.randn_like(std)
            z = mu + eps * std
        recon = vae.decode(z)
        X7_recon = unpack_image_to_7d(recon, meta)
        mse = F.mse_loss(X7_recon, X7)
    print(f"[Eval] MSE after fine-tuning = {mse.item():.6f}")


def eval_only(vae, X, device, tile=2, save_recon_path="./results/gaussians/recon_output.pth", comp=None):
    X7 = 2 * (X - X.min()) / (X.max() - X.min() + 1e-8) - 1
    vae = vae.to(device=device, dtype=torch.float32)
    vae.eval()

    x_img, meta = pack_7d_to_image(X7, tile)
    print(f"[Pack] -> {x_img.shape}")
    best_mse = 1
    with torch.no_grad():
        z = vae.encode(x_img)
        if isinstance(z, (tuple, list)):
            mu, logvar = z
            std = (0.5 * logvar).exp()
            eps = torch.randn_like(std)
            z = mu + eps * std
        print(f"[latent shape] {z.shape}")
        recon = vae.decode(z)
        X7_recon = unpack_image_to_7d(recon, meta)
        print(f"[recon] {X7_recon}")

        X7_recon_denorm = (X7_recon + 1) / 2 * (X.max() - X.min()) + X.min()
        xyz_recon = X7_recon_denorm[:, :2]
        fdc_recon = torch.round(X7_recon_denorm[:, 2:4]).clamp(min=1, max=8).to(torch.int)
        chol_recon = torch.round(X7_recon_denorm[:, 4:7]).clamp(min=0, max=63).to(torch.int).to(torch.float32)

        xyz_loss = F.mse_loss(xyz_recon, X[:, :2])
        fdc_loss = F.mse_loss(fdc_recon, X[:, 2:4])
        chol_loss = F.mse_loss(chol_recon, X[:, 4:7])
        print(f"[Eval] MSE (xyz) = {xyz_loss.item():.6f}")
        print(f"[Eval] MSE (fdc) = {fdc_loss.item():.6f}")
        print(f"[Eval] MSE (chol) = {chol_loss.item():.6f}")

        mse = F.mse_loss(X7_recon_denorm, X)
    print(f"[Eval] MSE (frozen VAE) = {mse.item():.6f}")

    # === 拆分回原始字段 ===
    xyz_recon = X7_recon_denorm[:, :2]
    fdc_recon = torch.round(X7_recon_denorm[:, 2:4]).clamp(min=1, max=8).to(torch.int)
    chol_recon = torch.round(X7_recon_denorm[:, 4:7]).clamp(min=0, max=63).to(torch.int).to(torch.float32)
    comp_recon = {
        "xyz": xyz_recon,
        "feature_dc_index": fdc_recon,
        "quant_cholesky_elements": chol_recon,
    }

    # === 保存重建结果 ===
    torch.save(comp_recon, save_recon_path)
    print(f"[Save] reconstructed data saved to {save_recon_path}")
    print(comp_recon)

    return comp_recon


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="comp_0.pth")
    parser.add_argument("--tile", type=int, default=2)
    parser.add_argument("--mode", type=str, default="eval", choices=["eval", "train"])
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--zdim", type=int, default=64)
    parser.add_argument("--dim", type=int, default=96)
    parser.add_argument("--ckpt", type=str, default="vae_tile_train.py")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === 载入 Gaussian 数据 ===
    comp = torch.load(Path(args.data))
    file_name = Path(args.data).stem
    save_path = Path(f"./results/gaussians/{file_name}_recon.pth")

    print(f"[gt] {comp}")
    xyz = comp["xyz"].to(device, dtype=torch.float32)
    fdc = comp["feature_dc_index"].to(device, dtype=torch.float32)
    chol = comp["quant_cholesky_elements"].to(device, dtype=torch.float32)
    X7 = torch.cat([xyz, fdc, chol], dim=-1)
    # X7 = 2 * (X7 - X7.min()) / (X7.max() - X7.min() + 1e-8) - 1
    print(f"[Data Size] {X7.shape}")

    # === 初始化 IV-VAE ===
    vae = IV_VAE(args.zdim, args.dim)
    vae.to(device=device, dtype=torch.float32)
    print(f"[Init] IV-VAE ready (z_dim={args.zdim}, dim={args.dim})")

    if args.mode == "eval":
        ckpt = 'ckpt/vae_tile_trained_comp_24_9000.pth'
        vae.load_state_dict(torch.load(ckpt, map_location=device))

    # === 模式选择 ===
    if args.mode == "train":
        train_and_eval(vae, X7, device, tile=args.tile, epochs=args.epochs, lr=args.lr)
    else:
        eval_only(vae, X7, device, tile=args.tile, save_recon_path=save_path)


if __name__ == "__main__":
    main()