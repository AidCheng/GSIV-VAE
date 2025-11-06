import torch
import torch.nn as nn

# 引入主模型骨架
from .vae3d import VAutoencoder3d

class IV_VAE(nn.Module):
    """
    自定义潜空间大小的 IV-VAE，用于单数据集 Overfitting。
    - 不依赖预训练权重
    - scale/shift 固定为 1.0 / 0.0
    - 允许自由设置 z_dim, dim, dropout 等参数
    """
    def __init__(
        self,
        z_dim=16,             # 潜变量维度（原版为16）
        dim=96,               # 通道宽度（主干宽度）
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0,
    ):
        super().__init__()

        # 配置项
        self.z_dim = z_dim
        self.dim = dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.dropout = dropout

        # 构建 3D VAE 主体
        self.model = VAutoencoder3d(
            dim=dim,
            z_dim=z_dim,
            dim_mult=dim_mult,
            num_res_blocks=num_res_blocks,
            attn_scales=attn_scales,
            temperal_downsample=temperal_downsample,
            dropout=dropout,
            scale_factor=1.0,   # 不再使用预训练scale
            shift_factor=0.0,
        )

    def forward(self, x):
        return self.model(x)

    def encode(self, x):
        return self.model.encode(x)

    def decode(self, z):
        return self.model.decode(z)

    def sample(self, mu, log_var, deterministic=False):
        return self.model.sample(mu, log_var, deterministic)

    def clear_cache(self):
        self.model.clear_cache() 
