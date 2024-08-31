import math
from typing import Any, Optional, Sequence, Tuple, Union

import torch
from einops import rearrange
from torch import Tensor, nn

from src.models._base_model import BaseModel
from src.models.foldingtrunk import FoldingTrunk
from src.models.structure_module import StructureModule
from src.models.modules.misc import get_time_embedder
from src.utilities.utils import exists, FAPEloss, RMSDloss


class DistributionalGraphormerInterpolation(BaseModel):
    def __init__(self, dropout=0.15, **kwargs):
        super().__init__(**kwargs)
        self.pro_esm_combine = nn.Parameter(torch.zeros(36 + 1))
        self.pro_esm_mlp = nn.Sequential(
            nn.LayerNorm(2560),
            nn.Linear(2560, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
        )
        self.sm_esm_mlp = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
        )

        self.ss2single = nn.Linear(1024, 384, bias=True)
        self.trunk = FoldingTrunk()
        self.diff_str = MainModel(dropout=dropout)

    def prepare_samples(self, inputs: dict) -> dict:
        past_steps = dict()
        last_step = dict()
        for key, value in inputs.items():
            past_steps[key] = value[0]
            last_step[key] = value[1]

        # condition
        aatype = torch.cat(
            [last_step['pro_aatype'], last_step['sm_aatype']], dim=1)
        mask = torch.cat([last_step['pro_mask'], last_step['sm_mask']], dim=1)

        pro_esm = (self.pro_esm_combine.softmax(0).unsqueeze(0)
                   @ last_step['pro_esm']).squeeze(2)
        pro_esm = self.pro_esm_mlp(pro_esm)

        sm_esm = self.sm_esm_mlp(last_step['sm_esm'])
        esm = torch.cat([pro_esm, sm_esm], dim=1)

        single_repr, pair_repr = self.trunk(esm, aatype, mask)
        single_repr = self.ss2single(single_repr)

        # T, IR for past_steps and last step
        past_steps_T = torch.cat(
            [past_steps['pro_T'], past_steps['sm_T']], dim=1)
        past_steps_IR = torch.cat(
            [past_steps['pro_IR'], past_steps['sm_IR']], dim=1)

        last_step_T = torch.cat([last_step['pro_T'], last_step['sm_T']], dim=1)
        last_step_IR = torch.cat(
            [last_step['pro_IR'], last_step['sm_IR']], dim=1)

        ret = {
            'single_repr': single_repr,
            'pair_repr': pair_repr,
            'mask': mask,
            'past_steps_T': past_steps_T,
            'past_steps_IR': past_steps_IR,
            'last_step_T': last_step_T,
            'last_step_IR': last_step_IR
        }

        return ret

    def forward(self, inputs: dict, time: Tensor, **kwargs):
        inputs = self.prepare_samples(inputs)
        predictions = self.diff_str(inputs, time)
        return predictions, inputs['mask']

    def get_loss(
        self,
        inputs: dict,
        targets: dict,
        time: Tensor,
        metadata: Any = None,
        return_predictions: bool = False,
        **kwargs
    ):
        num_residue = inputs['pro_aatype'][0].size(dim=-1)
        num_ligand_atom = inputs['sm_aatype'][0].size(dim=-1)
        predictions, mask = self(inputs, time, **kwargs)  # mask: (b, n)
        targets_T = torch.cat([targets['pro_T'], targets['sm_T']], dim=1)
        targets_IR = torch.cat([targets['pro_IR'], targets['sm_IR']], dim=1)
        if isinstance(self.criterion, FAPEloss):
            loss = self.criterion(
                predictions,
                (targets_T, targets_IR),
                padding_mask=mask.unsqueeze(-1)
            )
        elif isinstance(self.criterion, RMSDloss):
            loss = self.criterion(
                predictions,
                (targets_T, targets_IR),
                num_residue=num_residue,
                num_ligand_atom=num_ligand_atom,
                padding_mask=mask
            )
        if return_predictions:
            return loss, predictions
        return loss


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(
        self,
        dim,
        max_period=10000,
    ):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.dummy = nn.Parameter(
            torch.empty(0, dtype=torch.float), requires_grad=False
        )  # to detect fp16

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(self.max_period) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(
            half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        embeddings = embeddings.to(self.dummy.dtype)
        return embeddings


class RelativePositionBias(nn.Module):
    def __init__(self, num_buckets=64, max_distance=256, out_dim=2):
        super(RelativePositionBias, self).__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(self.num_buckets, out_dim)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets, max_distance):
        num_buckets //= 2
        ret = (relative_position < 0).to(relative_position) * num_buckets
        relative_position = torch.abs(relative_position)
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        val_if_large = (
            max_exact
            + (
                torch.log(relative_position / max_exact)
                / math.log(max_distance / max_exact)
                * (num_buckets - max_exact)
            ).long()
        )
        val_if_large = torch.min(
            val_if_large, torch.full_like(val_if_large, num_buckets - 1)
        )

        ret += torch.where(is_small, relative_position, val_if_large)
        return ret

    def forward(self, relative_position):
        rp_bucket = self._relative_position_bucket(
            relative_position,
            num_buckets=self.num_buckets,
            max_distance=self.max_distance,
        )
        rp_bias = self.relative_attention_bias(rp_bucket)
        return rp_bias


class MainModel(nn.Module):
    def __init__(self, d_model=768, d_pair=256, n_layer=12, n_heads=32, dropout=0.15):
        super(MainModel, self).__init__()

        self.step_emb = SinusoidalPositionEmbeddings(dim=d_model)
        self.x1d_proj = nn.Sequential(
            nn.LayerNorm(384), nn.Linear(384, d_model, bias=False)
        )
        self.x2d_proj = nn.Sequential(
            nn.LayerNorm(128), nn.Linear(128, d_pair, bias=False)
        )
        self.rp_proj = RelativePositionBias(
            num_buckets=64, max_distance=128, out_dim=d_pair
        )

        self.st_module = StructureModule(
            d_pair=d_pair,
            n_layer=n_layer,
            d_model=d_model,
            n_head=n_heads,
            dim_feedforward=1024,
            dropout=dropout,
        )

    def forward_step(self,
                     past_steps_pose: tuple,
                     last_step_pose: tuple,
                     mask: Tensor,
                     step: Tensor,
                     single_repr: Tensor,
                     pair_repr: Tensor
                     ):
        x1d = self.x1d_proj(single_repr) + self.step_emb(step)[:, None]
        x2d = self.x2d_proj(pair_repr)
        past_steps_T, past_steps_IR = past_steps_pose
        last_step_T, last_step_IR = last_step_pose

        pos = torch.arange(past_steps_T.shape[1], device=x1d.device)
        pos = pos.unsqueeze(1) - pos.unsqueeze(0)

        x2d = x2d + self.rp_proj(pos)[None]

        bias = mask.to(past_steps_T.dtype).masked_fill(
            mask, float("-inf"))[:, None, :, None]
        bias = bias.permute(0, 3, 1, 2)

        T_eps, IR_eps = self.st_module(
            past_steps_pose, last_step_pose, x1d, x2d, bias)

        T_eps = torch.matmul(past_steps_IR.transpose(-1, -2),
                             T_eps.unsqueeze(-1)).squeeze(-1)
        return T_eps, IR_eps

    def forward(self, inputs: dict, time: Tensor):
        mask = inputs['mask'] == 0
        past_steps_T = inputs['past_steps_T']
        past_steps_IR = inputs['past_steps_IR']
        last_step_T = inputs['last_step_T']
        last_step_IR = inputs['last_step_IR']

        past_steps_T.masked_fill_(mask[..., None], 0.0)
        last_step_T.masked_fill_(mask[..., None], 0.0)
        past_steps_IR.masked_fill_(mask[..., None, None], 0.0)
        last_step_IR.masked_fill_(mask[..., None, None], 0.0)

        pred_T, pred_IR = self.forward_step(
            (past_steps_T, past_steps_IR),
            (last_step_T, last_step_IR),
            mask,
            time,
            inputs["single_repr"],
            inputs["pair_repr"],
        )

        pred_T.masked_fill_(mask[..., None], 0.0)
        pred_IR.masked_fill_(mask[..., None, None], 0.0)

        return (pred_T, pred_IR)
