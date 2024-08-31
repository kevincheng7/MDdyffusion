from __future__ import annotations

import math
import os
import re
import random
from os.path import join
from pathlib import Path
from typing import Any, Dict, Optional, List

import numpy as np
import torch
from einops import rearrange
from torch import Tensor
from torch.utils import data
from omegaconf import OmegaConf
from openbabel import openbabel

from src.datamodules.abstract_datamodule import BaseDataModule
from src.datamodules.datasets.physical_systems_benchmark import TrajectoryDataset
from src.datamodules.torch_datasets import MyTensorDataset
from src.utilities.utils import (
    get_logger,
    raise_error_if_invalid_type,
    raise_error_if_invalid_value,
    raise_if_invalid_shape,
)
from src.mol_frame.myframe import parse_mol, process_sdf

log = get_logger(__name__)


class MDTrajectoryDataset(data.Dataset):
    """
    Since loading the entire trajectory at once requires a lot of memory, this dataset generates 
    times and filter them in `__getitem___`. 
    `__getitem__` only returns (window + 2) frames.
    Should be used during the training phase of the interpolation network.
    """

    def __init__(self, base_dir: str, indices_merged: List, window: int, horizon: int):
        super().__init__()
        self.base_dir = base_dir
        self.window = window
        self.horizon = horizon
        # [[{},{},...,{}],[{},{},...,{}],...,[{},{},...,{}]]
        self.indices_merged = indices_merged

    @property
    def horizon_range(self) -> List[int]:
        # h = horizon
        # We use timesteps  w-l+1, ..., w-1, w+h to predict timesteps w, ..., w+h-1
        # interpolate between step t=0 and t=horizon
        return list(np.arange(1, self.horizon))

    def process_pdb(self, input_file: str):
        with open(input_file, 'r') as f:
            lines = f.readlines()

        aa2index = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS',
                    'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                    'LEU', 'LYS', 'MET', 'PHE', 'PRO',
                    'SER', 'THR', 'TRP', 'TYR', 'VAL']
        croods = []
        aatype = []
        for line in lines:
            if not 'ATOM' in line:
                continue
            info = line.split()
            if info[2] in ['C', 'CA', 'N']:
                croods.append([eval(info[6]), eval(info[7]), eval(info[8])])
            if info[2] == 'CA':
                aatype.append(aa2index.index(info[3]))
        croods = np.array(croods)
        aatype = np.array(aatype)
        L = croods.shape[0] // 3
        croods = croods.reshape(L, 3, 3)

        return aatype, croods

    def calc_rotate_imat(self, N, CA, C):
        p1 = N - CA
        epsilon = 1e-10

        norm_p1 = np.linalg.norm(p1, axis=-1, keepdims=True)
        norm_p1 = np.where(norm_p1 == 0, epsilon, norm_p1)
        x = p1 / norm_p1

        p2 = C - N

        inner_1 = np.matmul(np.expand_dims(p1, axis=1),
                            np.expand_dims(p1, axis=2))[:, :, 0]
        inner_2 = np.matmul(np.expand_dims(-p1, axis=1),
                            np.expand_dims(p2, axis=2))[:, :, 0]

        inner_2_safe = np.where(inner_2 == 0, epsilon, inner_2)
        alpha = inner_1 / inner_2_safe

        y = alpha * p2 + p1

        norm_y = np.linalg.norm(y, axis=-1, keepdims=True)
        norm_y = np.where(norm_y == 0, epsilon, norm_y)
        y = y / norm_y

        z = np.cross(x, y)

        mat = np.concatenate([x, y, z], axis=-1)
        mat = mat.reshape(*mat.shape[:-1], 3, 3)
        assert not np.isnan(np.sum(
            mat)), "the calculated rotation matrix contains NaN, please check the input file."
        return mat

    def __getitem__(self, index):
        key_name = ['pro_aatype', 'pro_mask', 'pro_esm', 'pro_T', 'pro_IR',
                    'sm_aatype', 'sm_mask', 'sm_esm', 'sm_IR', 'sm_T', 'meta_data']
        ret = {key: [] for key in key_name}
        possible_times = self.horizon_range  # (h,)
        t = possible_times[random.randint(0, len(possible_times) - 1)]  # (1,)
        # (window + 2) frames selected
        items = list(range(self.window)) + [self.window + t - 1, -1]
        selected_data = [self.indices_merged[index][item]
                         for item in items]  # [{},{},{}] for window == 1

        for data in selected_data:
            pro_aatype, pro_xyz = self.process_pdb(
                os.path.join(self.base_dir, data['PDB File']))
            pro_esm = np.load(
                self.base_dir + data['Npy-Pro Files'], allow_pickle=True)[0]
            pro_T = pro_xyz[:, 1, :]  # select CA atom
            pro_IR = self.calc_rotate_imat(
                pro_xyz[:, 0, :], pro_xyz[:, 1, :], pro_xyz[:, 2, :])  # (num-residues, 3, 3)
            pro_mask = np.ones_like(pro_aatype)

            sm_aatype, sm_IR, sm_T, sm_atom_frames, obmol_ligand = process_sdf(
                os.path.join(self.base_dir, data['SDF Files']))
            sm_esm = np.load(
                self.base_dir + data['Npy-LIG Files'], allow_pickle=True)[0]
            sm_mask = np.ones_like(sm_aatype)

            ret['pro_aatype'].append(pro_aatype)
            ret['pro_mask'].append(pro_mask)
            ret['pro_esm'].append(pro_esm)
            ret['pro_T'].append(pro_T)
            ret['pro_IR'].append(pro_IR)
            ret['sm_aatype'].append(sm_aatype)
            ret['sm_esm'].append(sm_esm)
            ret['sm_T'].append(sm_T)
            ret['sm_IR'].append(sm_IR)
            ret['sm_mask'].append(sm_mask)
            ret['meta_data'].append({
                'system': data['system'],
                'time': data['time']
            })
        
        result = dict()
        for key, value in ret.items():
            if key != 'meta_data':
                result[key] = np.swapaxes(np.stack(value), axis1=0, axis2=1)
            else:
                result[key] = value

        result['time'] = t

        return result

    def __len__(self):
        return len(self.indices_merged)


class FullMDTrajectoryDataset(MDTrajectoryDataset):
    """
    This dataset load the entire trajectory, no time information is generated.
    `__getitem__` returns (window + horizon) frames.
    Should be used during the validation/test/predict phase of the interpolation network.    
    """

    def __init__(self, base_dir: str, indices_merged: List, window: int, horizon: int):
        super().__init__(base_dir, indices_merged, window, horizon)

    def __getitem__(self, index):
        key_name = ['pro_aatype', 'pro_mask', 'pro_esm', 'pro_T', 'pro_IR',
                    'sm_aatype', 'sm_mask', 'sm_esm', 'sm_IR', 'sm_T', 'meta_data']
        ret = {key: [] for key in key_name}
        for data in self.indices_merged[index]:
            pro_aatype, pro_xyz = self.process_pdb(
                os.path.join(self.base_dir, data['PDB File']))
            pro_esm = np.load(
                self.base_dir + data['Npy-Pro Files'], allow_pickle=True)[0]
            pro_T = pro_xyz[:, 1, :]  # select CA atom
            pro_IR = self.calc_rotate_imat(
                pro_xyz[:, 0, :], pro_xyz[:, 1, :], pro_xyz[:, 2, :])  # (num-residues, 3, 3)
            pro_mask = np.ones_like(pro_aatype)

            sm_aatype, sm_IR, sm_T, sm_atom_frames, obmol_ligand = process_sdf(
                os.path.join(self.base_dir, data['SDF Files']))
            sm_esm = np.load(
                self.base_dir + data['Npy-LIG Files'], allow_pickle=True)[0]
            sm_mask = np.ones_like(sm_aatype)

            ret['pro_aatype'].append(pro_aatype)
            ret['pro_mask'].append(pro_mask)
            ret['pro_esm'].append(pro_esm)
            ret['pro_T'].append(pro_T)
            ret['pro_IR'].append(pro_IR)
            ret['sm_aatype'].append(sm_aatype)
            ret['sm_esm'].append(sm_esm)
            ret['sm_T'].append(sm_T)
            ret['sm_IR'].append(sm_IR)
            ret['sm_mask'].append(sm_mask)
            ret['meta_data'].append({
                'system': data['system'],
                'time': data['time']
            })

        result = dict()
        for key, value in ret.items():
            if key != 'meta_data':
                result[key] = np.swapaxes(np.stack(value), axis1=0, axis2=1)
            else:
                result[key] = value

        return result
