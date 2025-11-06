import os
# Help defragmentation of GPU memory
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from models.vae3d import IV_VAE
from decord import VideoReader, cpu
import torch
from fractions import Fraction
import os
from einops import rearrange
from torchvision.io import write_video
from torchvision import transforms
from fire import Fire

@torch.no_grad()
def main(
    video_path: str = './video/ori.mp4',
    save_path: str = 'video/gen.mp4',
    height: int = 720,
    width: int = 1280,
    z_dim: int = 8,
    dim: int = 64,
    frames_per_batch: int = 4,  # adjust smaller if needed
):
    # Choose device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize VAE model on GPU in bfloat16 for inference
    vae3d = IV_VAE(z_dim, dim).to(device=device, dtype=torch.bfloat16)
    vae3d.requires_grad_(False)
    vae3d.eval()

    # Prepare resize transform (CPU)
    transform = transforms.Compose([
        transforms.Resize(size=(height, width))
    ])

    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Load video frames into CPU RAM
    video_reader = VideoReader(video_path, ctx=cpu(0))
    # Use a Fraction for FPS to satisfy PyAV
    raw_fps = video_reader.get_avg_fps()
    fps = Fraction(raw_fps).limit_denominator()

    # Read all frames as numpy and convert to tensor
    video_np = video_reader.get_batch(list(range(len(video_reader)))).asnumpy()
    video = rearrange(torch.tensor(video_np), 't h w c -> t c h w')
    video = transform(video)
    video = rearrange(video, 't c h w -> c t h w').unsqueeze(0).to(torch.bfloat16)

    # Trim to multiple of frames_per_batch
    total_frames = video.shape[2]
    usable_frames = (total_frames // frames_per_batch) * frames_per_batch
    video = video / 127.5 - 1.0
    video = video[:, :, :usable_frames, :, :]
    print(f'Shape of input video on CPU: {video.shape}')

    # Process in small temporal chunks
    decoded_chunks = []
    for start in range(0, usable_frames, frames_per_batch):
        end = start + frames_per_batch
        # Slice on CPU and move chunk to GPU
        v_chunk = video[:, :, start:end, :, :].to(device)
        print(f'Encoding frames {start}–{end} (chunk {v_chunk.shape})')

        # Encode + decode on GPU
        latent_chunk  = vae3d.encode(v_chunk)
        decoded_chunk = vae3d.decode(latent_chunk)

        # Move output back to CPU and clear GPU memory
        decoded_chunks.append(decoded_chunk.cpu())
        del v_chunk, latent_chunk, decoded_chunk
        torch.cuda.empty_cache()

    # Reassemble full video on CPU
    results = torch.cat(decoded_chunks, dim=2)
    print(f'Decoded video shape on CPU: {results.shape}')

    # Convert to [T, H, W, C] uint8 frames
    results = rearrange(results.squeeze(0), 'c t h w -> t h w c')
    results = ((torch.clamp(results, -1.0, 1.0) + 1.0) * 127.5).to(dtype=torch.uint8)

    # Write video (PyAV expects a rational fps)
    write_video(save_path, results, fps=fps, options={'crf': '10'})
    print(f'Video written to {save_path}')

if __name__ == '__main__':
    Fire(main)
