"""
@file encoders.py
@brief Multi-modal encoders for the DRL scheduling architecture.

@details
Each observation modality is processed by a dedicated encoder, and the
resulting embeddings are concatenated into a single 464-D vector consumed
by the FusionHead.

@par Encoder outputs (from architecture diagram)
| Encoder                              | Input             | Output |
|--------------------------------------|-------------------|--------|
| CNN-SPPF (Factory Floor)             | 64×64×3           | 256-D  |
| CNN-SPPF (Scheduling Matrix)         | n×2m×3            | 128-D  |
| Global Context MLP                   | 10-D              | 32-D   |
| Distance Embed MLP                   | 64-D              | 32-D   |
| Event Flag Embed                     | 6-D               | 16-D   |
| **Total concatenation**              |                   | 464-D  |
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class SPPF(nn.Module):
    """@brief Spatial Pyramid Pooling – Fast (YOLOv5-style).

    @details
    Applies sequential max-pools at multiple kernel sizes with same-padding,
    then concatenates the original and pooled feature maps for multi-scale
    spatial feature extraction.  A 1×1 convolution reduces the channel count
    before pooling and expands it afterward.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 pool_sizes: List[int] = None):
        """@brief Construct the SPPF module.

        @param in_channels   Number of input feature-map channels.
        @param out_channels  Number of output feature-map channels after
                             the expand convolution.
        @param pool_sizes    List of max-pool kernel sizes (default [5, 9, 13]).
        """
        super().__init__()

        ## @brief Max-pool kernel sizes defining the spatial pyramid.
        self.pool_sizes = pool_sizes or [5, 9, 13]
        mid = in_channels // 2

        ## @brief 1×1 conv that halves channels before pooling.
        self.conv_reduce = nn.Sequential(
            nn.Conv2d(in_channels, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(inplace=True),
        )

        ## @brief Parallel max-pool branches with same-padding.
        self.pools = nn.ModuleList([
            nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
            for k in self.pool_sizes
        ])

        # 1 original + len(pool_sizes) pooled = 4 branches total
        concat_channels = mid * (1 + len(self.pool_sizes))

        ## @brief 1×1 conv that projects concatenated branches to @p out_channels.
        self.conv_expand = nn.Sequential(
            nn.Conv2d(concat_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """@brief Apply multi-scale pooling and concatenation.

        @param x  Feature map of shape (B, @p in_channels, H, W).
        @return Feature map of shape (B, @p out_channels, H, W).
        """
        x = self.conv_reduce(x)
        branches = [x]
        for pool in self.pools:
            branches.append(pool(x))
        x = torch.cat(branches, dim=1)
        return self.conv_expand(x)


class CNNSPPFEncoder(nn.Module):
    """@brief CNN backbone + SPPF head → global average pool → flat embedding.

    @details
    Three convolutional blocks (with BatchNorm, SiLU, and 2× max-pool
    down-sampling in the first two) feed into an SPPF module.  The
    resulting spatial features are globally average-pooled and projected
    to @p out_dim via a linear layer with LayerNorm.

    Used for both the Factory Floor grid and the Scheduling Matrix image.
    """

    def __init__(self, in_channels: int, out_dim: int,
                 pool_sizes: List[int] = None):
        """@brief Construct the CNN-SPPF encoder.

        @param in_channels  Number of input image channels (e.g. 3).
        @param out_dim      Dimensionality of the output embedding vector.
        @param pool_sizes   Kernel sizes forwarded to @ref SPPF.
        """
        super().__init__()

        ## @brief Lightweight three-block convolutional backbone.
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

        ## @brief SPPF module for multi-scale spatial aggregation.
        self.sppf = SPPF(128, 128, pool_sizes=pool_sizes)

        ## @brief Global average pool + linear projection to @p out_dim.
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """@brief Encode a spatial image into a flat embedding.

        @param x  Image tensor of shape (B, C, H, W).
        @return Embedding vector of shape (B, @p out_dim).
        """
        x = self.backbone(x)
        x = self.sppf(x)
        return self.head(x)


class MLPEncoder(nn.Module):
    """@brief Simple two-layer MLP encoder for vector inputs.

    @details
    Used for scalar features, pairwise distances, and event flags.
    Architecture: Linear → LayerNorm → SiLU → Linear → LayerNorm → SiLU.
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = None):
        """@brief Construct the MLP encoder.

        @param in_dim      Dimensionality of the input vector.
        @param out_dim     Dimensionality of the output embedding.
        @param hidden_dim  Width of the hidden layer.  Defaults to
                           max(@p out_dim × 2, @p in_dim).
        """
        super().__init__()
        hidden_dim = hidden_dim or max(out_dim * 2, in_dim)

        ## @brief Two-layer MLP with LayerNorm and SiLU activations.
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """@brief Encode a 1-D vector into a lower-dimensional embedding.

        @param x  Input vector of shape (B, @p in_dim).
        @return Embedding of shape (B, @p out_dim).
        """
        return self.net(x)


class MultiModalEncoder(nn.Module):
    """@brief Full encoder that processes all five observation modalities
    and concatenates the results into a 464-D vector.

    @details
    Each modality is handled by a dedicated sub-encoder:

    | Sub-encoder          | Observation key     | Input shape       | Output dim |
    |----------------------|---------------------|-------------------|------------|
    | @ref factory_encoder | @c factory_grid     | (B, 3, 64, 64)    | 256        |
    | @ref sched_encoder   | @c sched_matrix     | (B, 3, 100, 40)   | 128        |
    | @ref global_mlp      | @c global_scalars   | (B, 10)           | 32         |
    | @ref distance_mlp    | @c distance_matrix  | (B, 64)           | 32         |
    | @ref event_embed     | @c event_flags      | (B, 6)            | 16         |

    The five embeddings are concatenated along the feature dimension to
    produce a single (B, 464) tensor.
    """

    def __init__(self, encoder_cfg=None):
        """@brief Construct the multi-modal encoder.

        @param encoder_cfg  An EncoderConfig dataclass specifying output
                            dimensions for each sub-encoder.  Defaults to
                            EncoderConfig() if None.
        """
        super().__init__()
        from config import EncoderConfig
        cfg = encoder_cfg or EncoderConfig()

        ## @brief CNN-SPPF encoder for the factory-floor occupancy grid (→ 256-D).
        self.factory_encoder = CNNSPPFEncoder(
            in_channels=3,
            out_dim=cfg.factory_cnn_out,       # 256
            pool_sizes=cfg.sppf_pool_sizes,
        )

        ## @brief CNN-SPPF encoder for the scheduling-matrix image (→ 128-D).
        self.sched_encoder = CNNSPPFEncoder(
            in_channels=3,
            out_dim=cfg.sched_cnn_out,         # 128
            pool_sizes=cfg.sppf_pool_sizes,
        )

        ## @brief MLP encoder for normalized global scalar features (→ 32-D).
        self.global_mlp = MLPEncoder(
            in_dim=10,
            out_dim=cfg.global_mlp_out,        # 32
        )

        ## @brief MLP encoder for flattened pairwise distance matrix (→ 32-D).
        self.distance_mlp = MLPEncoder(
            in_dim=64,
            out_dim=cfg.distance_mlp_out,      # 32
        )

        ## @brief MLP encoder for binary event flags (→ 16-D).
        self.event_embed = MLPEncoder(
            in_dim=6,
            out_dim=cfg.event_embed_out,       # 16
        )

        ## @brief Total concatenated output dimensionality (464).
        self.output_dim = cfg.concat_dim       # 464

    def forward(self, obs: dict) -> torch.Tensor:
        """@brief Encode all observation modalities and concatenate.

        @param obs  Dict with keys matching the state-space components:
                    @c factory_grid, @c sched_matrix, @c global_scalars,
                    @c distance_matrix, and @c event_flags.
        @return Concatenated multi-modal embedding of shape
                (B, @ref output_dim).
        """
        h_factory = self.factory_encoder(obs["factory_grid"])
        h_sched = self.sched_encoder(obs["sched_matrix"])
        h_global = self.global_mlp(obs["global_scalars"])
        h_dist = self.distance_mlp(obs["distance_matrix"])
        h_event = self.event_embed(obs["event_flags"])

        return torch.cat(
            [h_factory, h_sched, h_global, h_dist, h_event], dim=-1
        )