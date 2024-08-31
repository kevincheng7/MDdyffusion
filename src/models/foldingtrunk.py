import typing as T
from contextlib import ExitStack
from dataclasses import dataclass
import numpy as np

import torch
import torch.nn as nn
from torch.nn import LayerNorm

from esm.esmfold.v1.tri_self_attn_block import TriangularSelfAttentionBlock
from esm.esmfold.v1.trunk import RelativePosition


@dataclass
class FoldingTrunkConfig:
    _name: str = "FoldingTrunkConfig"
    num_blocks: int = 4
    sequence_state_dim: int = 1024
    pairwise_state_dim: int = 128
    sequence_head_width: int = 32
    pairwise_head_width: int = 32
    position_bins: int = 32
    dropout: float = 0
    layer_drop: float = 0
    cpu_grad_checkpoint: bool = False

    max_recycles: int = 4
    chunk_size: T.Optional[int] = None


class FoldingTrunk(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.cfg = FoldingTrunkConfig(**kwargs)

        self.device_param = nn.Parameter(torch.zeros(1))

        c_s = self.cfg.sequence_state_dim
        c_z = self.cfg.pairwise_state_dim

        self.n_tokens_embed = 80 + 3
        self.pad_idx = 0
        self.unk_idx = self.n_tokens_embed - 2
        self.mask_idx = self.n_tokens_embed - 1
        self.embedding = nn.Embedding(self.n_tokens_embed, c_s, padding_idx=0)

        block = TriangularSelfAttentionBlock

        self.pairwise_positional_embedding = RelativePosition(
            self.cfg.position_bins, c_z)

        self.blocks = nn.ModuleList(
            [
                block(
                    sequence_state_dim=c_s,
                    pairwise_state_dim=c_z,
                    sequence_head_width=self.cfg.sequence_head_width,
                    pairwise_head_width=self.cfg.pairwise_head_width,
                    dropout=self.cfg.dropout,
                )
                for i in range(self.cfg.num_blocks)
            ]
        )

        self.chunk_size = self.cfg.chunk_size

    def forward(
        self,
        s_s_0,
        aa: torch.Tensor,
        mask: T.Optional[torch.Tensor] = None,
        residx: T.Optional[torch.Tensor] = None,
    ):

        if mask is None:
            mask = torch.ones_like(aa)

        B = aa.shape[0]
        L = aa.shape[1]
        device = aa.device

        if residx is None:
            residx = torch.arange(L, device=device).expand_as(aa)

        s_s_0 += self.embedding(aa)
        s_z_0 = s_s_0.new_zeros(B, L, L, self.cfg.pairwise_state_dim)

        def trunk_iter(s, z, residx, mask):
            z = z + self.pairwise_positional_embedding(residx, mask=mask)

            for block in self.blocks:
                s, z = block(s, z, mask=mask, residue_index=residx,
                             chunk_size=self.chunk_size)
                # torch.Size([2, 390, 1024]) torch.Size([2, 390, 390, 128])
            return s, z

        # pose_bins = FoldingTrunk.distogram(pose[-1][:, :, :3], 3.375, 21.375, self.recycle_bins)
        # pose_z += self.recycle_disto(recycle_bins.detach())

        s_s, s_z = trunk_iter(s_s_0, s_z_0, residx, mask)

        return s_s, s_z

    @property
    def device(self):
        return self.device_param.device
