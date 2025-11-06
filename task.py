import os
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from fire import Fire
from models.vae3d import IV_VAE  # Replace with 2D VAE if needed

@torch.no_grad()
def main(
    image_path: str = './image/input.png',      # PNG input
    save_path: str = './image/output.png',      # PNG output
    height: int = 720,
    width: int = 1280,
    z_dim: int = 8,
    dim: int = 64,
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    vae3d = IV_VAE(z_dim, dim).to(device=device, dtype=torch.bfloat16)
    vae3d.requires_grad_(False)
    vae3d.eval()

    # Load and preprocess PNG image
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),  # [0, 1]
        transforms.Lambda(lambda x: x * 2 - 1)  # to [-1, 1]
    ])
    img_tensor = transform(image).unsqueeze(0).to(device=device, dtype=torch.bfloat16)  # [1, 3, H, W]

    # Add time dimension for 3D VAE
    img_tensor = img_tensor.unsqueeze(2)  # [1, 3, 1, H, W]

    # Forward pass
    latent = vae3d.encode(img_tensor)
    output = vae3d.decode(latent)

    # Postprocess and save
    output = output.squeeze(0).squeeze(1)  # [3, H, W]
    output = torch.clamp((output + 1) / 2, 0, 1)  # back to [0,1]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_image(output.to(dtype=torch.float32), save_path)
    print(f'PNG image written to {save_path}')

if __name__ == '__main__':
    Fire(main)