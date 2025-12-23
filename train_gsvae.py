import torch
import matplotlib.pyplot as plt
import argparse
import math
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from models.guassian_iv_vae import IV_VAE 
from types import SimpleNamespace
from models.guassian_iv_vae import IV_VAE
import io, contextlib
import sys
import os

# zdim = 16, dim = 96
data_type = ['xyz', 'feature_dc_index', 'quant_cholesky_elements']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PatchDataset(Dataset):
    def __init__(self, data, device):
        """
        data: Tensor of shape (N, 3, 1, H, W)
        """
        self.data = data
        self.device = device

    def __len__(self): 
        return self.data.shape[0]

    def __getitem__(self, index):
        x = self.data[index].to(device=self.device, dtype=torch.float32)
        return x

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Train IV_VAE")
    parser.add_argument('--dataset', type=str, required=True, help='Path to directory containing .pth patches')
    parser.add_argument('--output', type=str, required=True, help='Path to save output models')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--z_dim', type=int, default=16)
    parser.add_argument('--dim', type=int, default=96)
    parser.add_argument('--enable_modifier', action='store_true')
    
    # If argv is None, argparse reads from sys.argv automatically
    return parser.parse_args(argv if argv is not None else sys.argv[1:])

def init_vae(args):
    # Suppress potential printing during init
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        # Ensure your IV_VAE accepts the args correctly
        vae = IV_VAE(args.z_dim, args.dim).to(device=device, dtype=torch.float32)
    return vae

def preprocess(data):
    """
    Removed H, W from arguments as they were hardcoded inside.
    """
    result = []
    H = 32 
    W = 64
    
    for X in data: 
        X_processed = X
        # Handle 1D tensors (N,) -> (N, 1)
        if len(X.shape) == 1:
            X_processed = X.unsqueeze(1)
            
        if X_processed.shape[1] < 3:
            pad = torch.zeros(X_processed.shape[0], 3 - X_processed.shape[1], device=X.device)
            X_processed = torch.cat([X_processed, pad], dim=1) 
        
        N, C = X_processed.shape

        pad_length = H * W - N
        if pad_length > 0:
            X_processed = torch.cat(
                [X_processed, torch.zeros(pad_length, C, device=X.device)], dim=0
            )
        elif pad_length < 0:
            # Simple truncation if data is larger than H*W
             X_processed = X_processed[:H*W, :]

        # reshape to [1, C, 1, H, W]
        # Transpose (N, C) -> (C, N) then view
        X_processed = X_processed.T.view(C, 1, H, W)
        result.append(X_processed.unsqueeze(0)) # shape: (1, C, 1, H, W)
    
    return result
    
def train(model, input_data, output_path, mode, cfg):
    """
    Added 'mode' argument usage and model saving.
    """
    loader = DataLoader(
        input_data,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    model.train()
    loss_log = []
    best_loss = float('inf')
    
    print(f"--- Starting Training for Mode: {mode} ---")

    for epoch in range(cfg.epochs):
        epoch_loss = 0
        count = 0
        for x in loader:
            optimizer.zero_grad()

            z = model.encode(x)
            x_recon = model.decode(z)

            # if mode == 'xyz':
            #     loss = chamfer_distance(x, x_recon)
            # else:
            loss = torch.nn.functional.mse_loss(x_recon, x)

            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            count += 1
        
        avg_loss = epoch_loss / count if count > 0 else 0

        if avg_loss < best_loss:
            best_loss = avg_loss
            # Create output directory if not exists
            os.makedirs(output_path, exist_ok=True)
            # Save the best model
            torch.save(model.state_dict(), f"{output_path}/best_model_{mode}.pth")
            
        if epoch % 10 == 0:
            print(f"[Train] mode: {mode} | epoch {epoch:4d} | loss={avg_loss:.6f}")
            loss_log.append(avg_loss)

    # Save loss log
    save_path = cfg.log_path / f"loss_log_{mode}.txt"
    if not cfg.log_path.exists():
        cfg.log_path.mkdir(parents=True)

    x = list(range(len(loss_log)))
    y = loss_log

    # 画图
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, linewidth=2)
    plt.xlabel("Step" if mode == "train" else "Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve ({mode})")
    plt.grid(True)

    # 保存
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    model.eval()
    test_data = next(iter(loader))
    with torch.no_grad():
        z = model.encode(test_data)
        recon = model.decode(z)
        test_loss = torch.nn.functional.mse_loss(recon, test_data).item()
    print(f"[Eval] mode: {mode} | Test Loss = {test_loss:.6f}")


def main(argv=None):
    args = parse_args(argv)
    data_path = args.dataset
    output_path = args.output
    
    # Ensure data path is a Path object
    patch_files = sorted(Path(data_path).glob('*.pth'))
    
    if not patch_files:
        print(f"No .pth files found in {data_path}")
        return

    xyz = []
    feat = []
    chol = []

    print("Preprocessing data...")
    for p in patch_files:
        patch_data = torch.load(p, map_location=device)
        # Pass only the list comprehension, preprocess handles the logic
        processed = preprocess([patch_data[d] for d in data_type])

        xyz.append(processed[0])
        feat.append(processed[1])
        chol.append(processed[2])

    # Stack list of tensors into a single tensor
    xyz = PatchDataset(torch.cat(xyz, dim=0), device=device)
    feat = PatchDataset(torch.cat(feat, dim=0), device=device)
    chol = PatchDataset(torch.cat(chol, dim=0), device=device)

    cfg = SimpleNamespace(
        lr = args.lr,
        batch_size = args.batch_size, 
        epochs = args.epochs,
        enable_modifier = args.enable_modifier,
        log_path = Path.cwd() / "logs",
    )

    input_datasets = [xyz, feat, chol]
    modes = ['xyz', 'feat', 'chol'] 

    for i in range(3):
        print(f"\nInitializing VAE for {modes[i]}...")
        vae = init_vae(args)
        # Fixed: passed 'modes[i]' to the train function
        train(vae, input_datasets[i], output_path, modes[i], cfg)

if __name__ == "__main__":
    main()