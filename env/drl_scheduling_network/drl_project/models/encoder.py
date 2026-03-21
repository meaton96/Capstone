"""
Multi-modal encoders for the DRL scheduling architecture.

Encoder outputs (from diagram):
  - CNN-SPPF (Factory Floor 64x64x3)   -> 256-D
  - CNN-SPPF (Scheduling Matrix n×2m×3) -> 128-D
  - Global Context MLP (10-D)           -> 32-D
  - Distance Embed MLP (64-D)           -> 32-D
  - Event Flag Embed (6-D)              -> 16-D
  - Total concat                        -> 464-D
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class SPPF(nn.Module):
    """
    Spatial Pyramid Pooling - Fast (YOLOv5-style).

    Applies sequential max-pools at multiple kernel sizes, then
    concatenates the results for multi-scale spatial feature extraction.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 pool_sizes: List[int] = None):
        super().__init__()
        self.pool_sizes = pool_sizes or [5, 9, 13]
        mid = in_channels // 2

        self.conv_reduce = nn.Sequential(
            nn.Conv2d(in_channels, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(inplace=True),
        )

        # Each pool branch feeds same-sized feature maps (same padding)
        self.pools = nn.ModuleList([
            nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
            for k in self.pool_sizes
        ])

        # 1 original + len(pool_sizes) pooled = 4 branches total
        concat_channels = mid * (1 + len(self.pool_sizes))
        self.conv_expand = nn.Sequential(
            nn.Conv2d(concat_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_reduce(x)
        branches = [x]
        for pool in self.pools:
            branches.append(pool(x))
        x = torch.cat(branches, dim=1)
        return self.conv_expand(x)


class CNNSPPFEncoder(nn.Module):
    """
    CNN backbone + SPPF head -> global average pool -> flat embedding.

    Used for both the Factory Floor grid and Scheduling Matrix image.
    """

    def __init__(self, in_channels: int, out_dim: int,
                 pool_sizes: List[int] = None):
        super().__init__()

        # Lightweight conv backbone
        self.backbone = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
        )

        self.sppf = SPPF(128, 128, pool_sizes=pool_sizes)

        # Global average pool + projection to target dim
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor
        Returns:
            (B, out_dim) embedding
        """
        x = self.backbone(x)
        x = self.sppf(x)
        return self.head(x)


class MLPEncoder(nn.Module):
    """Simple MLP encoder for vector inputs (scalars, distances, flags)."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = None):
        super().__init__()
        hidden_dim = hidden_dim or max(out_dim * 2, in_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiModalEncoder(nn.Module):
    """
    Full encoder that processes all 5 observation modalities
    and concatenates into a 464-D vector.

    Inputs:
        factory_grid:    (B, 3, 64, 64)  - spatial occupancy grid
        sched_matrix:    (B, 3, 100, 40) - scheduling matrix image
        global_scalars:  (B, 10)         - normalized scalar features
        distance_matrix: (B, 64)         - flattened 8x8 pairwise distances
        event_flags:     (B, 6)          - binary event indicators
    """

    def __init__(self, encoder_cfg=None):
        super().__init__()
        from config import EncoderConfig
        cfg = encoder_cfg or EncoderConfig()

        self.factory_encoder = CNNSPPFEncoder(
            in_channels=3,
            out_dim=cfg.factory_cnn_out,       # 256
            pool_sizes=cfg.sppf_pool_sizes,
        )
        self.sched_encoder = CNNSPPFEncoder(
            in_channels=3,
            out_dim=cfg.sched_cnn_out,         # 128
            pool_sizes=cfg.sppf_pool_sizes,
        )
        self.global_mlp = MLPEncoder(
            in_dim=10,
            out_dim=cfg.global_mlp_out,        # 32
        )
        self.distance_mlp = MLPEncoder(
            in_dim=64,
            out_dim=cfg.distance_mlp_out,      # 32
        )
        self.event_embed = MLPEncoder(
            in_dim=6,
            out_dim=cfg.event_embed_out,       # 16
        )

        self.output_dim = cfg.concat_dim       # 464

    def forward(self, obs: dict) -> torch.Tensor:
        """
        Args:
            obs: dict with keys matching state space components
        Returns:
            (B, 464) concatenated multi-modal embedding
        """
        h_factory = self.factory_encoder(obs["factory_grid"])
        h_sched = self.sched_encoder(obs["sched_matrix"])
        h_global = self.global_mlp(obs["global_scalars"])
        h_dist = self.distance_mlp(obs["distance_matrix"])
        h_event = self.event_embed(obs["event_flags"])

        return torch.cat(
            [h_factory, h_sched, h_global, h_dist, h_event], dim=-1
        )
