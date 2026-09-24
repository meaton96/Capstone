"""
@file encoders.py
@brief Multi-modal encoders for the DRL scheduling architecture.

@details
Each observation modality is processed by a dedicated encoder, and the
resulting embeddings are concatenated into a single 560-D vector consumed
by the FusionHead.

@par Encoder outputs (observation schema v2)
| Encoder                              | Input             | Output |
|--------------------------------------|-------------------|--------|
| CNN-SPPF (Factory Floor)             | 64×64×3           | 256-D  |
| Set encoder (Machine table)          | 100×16            | 128-D  |
| Set encoder (Job table)              | 64×17             | 128-D  |
| Global Context MLP                   | 16-D              | 32-D   |
| Event Flag Embed                     | 6-D               | 16-D   |
| **Total concatenation**              |                   | 560-D  |

v1 used a CNN over an 8-machine scheduling matrix and an MLP over an 8×8 distance matrix; both were
replaced because they could not represent the 15-machine floor (see config.py).
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




def _masked_pool(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """@brief Masked mean and max over the row axis.

    @param h     Row embeddings of shape (B, R, D).
    @param mask  Row mask of shape (B, R), 1 for rows to pool over.
    @return (B, 2D): mean then max. Both are 0 when no row is selected.
    """
    m = mask.unsqueeze(-1)
    count = m.sum(dim=1).clamp(min=1.0)
    mean = (h * m).sum(dim=1) / count
    very_neg = torch.finfo(h.dtype).min
    mx = h.masked_fill(m == 0, very_neg).max(dim=1).values
    mx = torch.where(mask.sum(dim=1, keepdim=True) > 0, mx, torch.zeros_like(mx))
    return torch.cat([mean, mx], dim=-1)


class SetEncoder(nn.Module):
    """@brief Permutation-invariant encoder for an entity table (machines or jobs).

    @details
    A shared two-layer MLP embeds every row; rows are then pooled twice with masked mean + max:
    once over all present rows (floor / WIP state) and once over the rows flagged as options in
    the current decision (the candidate machines of a routing decision, or the queue of a dispatch
    decision). The four pooled vectors are projected to @p out_dim.

    Because pooling ignores row count and order, the same weights apply to any floor up to the
    table size: this is what lets a policy trained on 15 machines be evaluated on 30-100.
    """

    def __init__(self, in_features: int, out_dim: int, hidden: int = 64,
                 present_col: int = 0, candidate_col: int = None):
        """@brief Construct the set encoder.

        @param in_features    Features per row.
        @param out_dim        Output embedding size.
        @param hidden         Per-row embedding size.
        @param present_col    Column holding the present mask (1 real row, 0 padding).
        @param candidate_col  Column flagging decision candidates, or None to skip that pool.
        """
        super().__init__()
        self.present_col = present_col
        self.candidate_col = candidate_col
        ## @brief Shared per-row embedding.
        self.row_mlp = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.SiLU(inplace=True),
        )
        n_pools = 2 if candidate_col is not None else 1
        ## @brief Projection of the pooled (mean, max) vectors.
        self.proj = nn.Sequential(
            nn.Linear(2 * hidden * n_pools, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, table: torch.Tensor) -> torch.Tensor:
        """@brief Encode a batch of tables.

        @param table  Shape (B, R, F).
        @return Shape (B, @p out_dim).
        """
        present = (table[..., self.present_col] > 0.5).to(table.dtype)
        h = self.row_mlp(table)
        pooled = [_masked_pool(h, present)]
        if self.candidate_col is not None:
            cand = present * (table[..., self.candidate_col] > 0.5).to(table.dtype)
            pooled.append(_masked_pool(h, cand))
        return self.proj(torch.cat(pooled, dim=-1))


class MultiModalEncoder(nn.Module):
    """@brief Full encoder that processes all five observation modalities
    and concatenates the results into a 560-D vector.

    @details
    Each modality is handled by a dedicated sub-encoder:

    | Sub-encoder          | Observation key     | Input shape       | Output dim |
    |----------------------|---------------------|-------------------|------------|
    | @ref factory_encoder | @c factory_grid     | (B, 3, 64, 64)    | 256        |
    | @ref machine_encoder | @c machine_table    | (B, 100, 16)      | 128        |
    | @ref job_encoder     | @c job_table        | (B, 64, 17)       | 128        |
    | @ref global_mlp      | @c global_scalars   | (B, 16)           | 32         |
    | @ref event_embed     | @c event_flags      | (B, 6)            | 16         |
    """

    def __init__(self, encoder_cfg=None):
        """@brief Construct the multi-modal encoder.

        @param encoder_cfg  An EncoderConfig dataclass specifying output
                            dimensions for each sub-encoder.  Defaults to
                            EncoderConfig() if None.
        """
        super().__init__()
        from config import (
            EncoderConfig, MACHINE_FEATURES, JOB_FEATURES, GLOBAL_SCALARS, EVENT_FLAGS,
            MACHINE_PRESENT_COL, MACHINE_CANDIDATE_COL, JOB_PRESENT_COL, JOB_CANDIDATE_COL,
        )
        cfg = encoder_cfg or EncoderConfig()

        ## @brief CNN-SPPF encoder for the factory-floor occupancy grid (→ 256-D).
        self.factory_encoder = CNNSPPFEncoder(
            in_channels=3,
            out_dim=cfg.factory_cnn_out,
            pool_sizes=cfg.sppf_pool_sizes,
        )

        ## @brief Set encoder for the machine table (→ 128-D).
        self.machine_encoder = SetEncoder(
            MACHINE_FEATURES, cfg.machine_set_out, cfg.set_row_hidden,
            present_col=MACHINE_PRESENT_COL, candidate_col=MACHINE_CANDIDATE_COL,
        )

        ## @brief Set encoder for the active-job table (→ 128-D).
        self.job_encoder = SetEncoder(
            JOB_FEATURES, cfg.job_set_out, cfg.set_row_hidden,
            present_col=JOB_PRESENT_COL, candidate_col=JOB_CANDIDATE_COL,
        )

        ## @brief MLP encoder for global scalar features (→ 32-D).
        self.global_mlp = MLPEncoder(in_dim=GLOBAL_SCALARS, out_dim=cfg.global_mlp_out)

        ## @brief MLP encoder for binary event flags (→ 16-D).
        self.event_embed = MLPEncoder(in_dim=EVENT_FLAGS, out_dim=cfg.event_embed_out)

        ## @brief Total concatenated output dimensionality (560).
        self.output_dim = cfg.concat_dim

    def forward(self, obs: dict) -> torch.Tensor:
        """@brief Encode all observation modalities and concatenate.

        @param obs  Dict with keys @c factory_grid, @c machine_table, @c job_table,
                    @c global_scalars and @c event_flags.
        @return Concatenated multi-modal embedding of shape (B, @ref output_dim).
        """
        return torch.cat([
            self.factory_encoder(obs["factory_grid"]),
            self.machine_encoder(obs["machine_table"]),
            self.job_encoder(obs["job_table"]),
            self.global_mlp(obs["global_scalars"]),
            self.event_embed(obs["event_flags"]),
        ], dim=-1)
