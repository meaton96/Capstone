"""
@file future_net.py
@brief SchedulingNetwork with a future-job encoder (2026-10-15, rq2-oracle-bc-future): a second set encoder over the
       future job table (des_twin.future_table, obs key "future_job_table"), concatenated with the five state
       embeddings before the fusion head.

@details The live job table's set encoder (models.encoder.SetEncoder: shared row MLP, masked mean + max pool,
projection) with its own weights, since the future rows have their own columns. Its 128-D output joins the 560-D
state embedding, so the fusion head is 688 -> 512 -> 256 and can mix the future with the current state (the 15
look-ahead features of rq3-lookahead were appended after the fusion head instead). Everything after the fusion head
is SchedulingNetwork's, including the optional policy / critic look-ahead inputs.

Built only from names that exist in every v5 code snapshot (config, models.encoder, models.actor_critic,
models.network), so an experiment can load this file on top of a frozen snapshot (results/<name>/code).
"""
import torch

from config import FusionConfig
from models.actor_critic import FusionHead
from models.encoder import SetEncoder
from models.network import SchedulingNetwork

FUTURE_OBS_KEY = "future_job_table"
FUTURE_SET_OUT = 128


class FutureSchedulingNetwork(SchedulingNetwork):
    """@brief SchedulingNetwork whose fused features also encode obs["future_job_table"] (rows, features)."""

    def __init__(self, encoder_cfg=None, fusion_cfg=None, ac_cfg=None, future_features: int = None,
                 future_out: int = FUTURE_SET_OUT):
        super().__init__(encoder_cfg, fusion_cfg, ac_cfg)
        if future_features is None:
            from des_twin.future_table import FUTURE_JOB_FEATURES
            future_features = FUTURE_JOB_FEATURES
        from des_twin.future_table import FUTURE_PRESENT_COL
        fusion_cfg = fusion_cfg or FusionConfig()
        self.future_features, self.future_out = int(future_features), int(future_out)
        ## @brief Set encoder for the future job table (-> future_out).
        self.future_encoder = SetEncoder(self.future_features, self.future_out, self.encoder_cfg.set_row_hidden,
                                         present_col=FUTURE_PRESENT_COL, candidate_col=None)
        ## @brief Fusion over the state embedding and the future embedding.
        self.fusion = FusionHead(input_dim=self.encoder_cfg.concat_dim + self.future_out,
                                 hidden_dim=fusion_cfg.hidden_dim, output_dim=fusion_cfg.output_dim)

    def _features(self, obs: dict) -> torch.Tensor:
        table = obs.get(FUTURE_OBS_KEY)
        if table is None:
            raise ValueError(f"this policy reads the future job table (obs '{FUTURE_OBS_KEY}'), but the observation has "
                             "none: build it with des_twin.future_table on the twin")
        fused = self.fusion(torch.cat([self.encoder(obs), self.future_encoder(table.to(torch.float32))], dim=-1))
        if self.ac_cfg.policy_extra_dim:
            extra = obs.get("critic_lookahead")
            if extra is None:
                raise ValueError("this policy reads look-ahead features (obs 'critic_lookahead'), but the observation "
                                 "has none: train / evaluate with --policy-lookahead on the twin")
            fused = torch.cat([fused, extra.to(fused.dtype)], dim=-1)
        return fused


def reads_future_table(state_dict: dict) -> bool:
    """@brief Whether a saved model has a future-job encoder (load it as FutureSchedulingNetwork)."""
    return any(k.startswith("future_encoder.") for k in state_dict)


def future_features_of(state_dict: dict) -> int:
    """@brief Columns per future row of a saved FutureSchedulingNetwork (its row MLP's input width)."""
    return int(state_dict["future_encoder.row_mlp.0.weight"].shape[1])


def network_for(state_dict: dict, encoder_cfg=None, ac_cfg=None):
    """@brief A fresh network of the right class for a saved state dict (load the weights into it yourself)."""
    if reads_future_table(state_dict):
        return FutureSchedulingNetwork(encoder_cfg, None, ac_cfg, future_features=future_features_of(state_dict),
                                       future_out=int(state_dict["future_encoder.proj.0.bias"].shape[0]))
    return SchedulingNetwork(encoder_cfg, None, ac_cfg)
