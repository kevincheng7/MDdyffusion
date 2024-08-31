from __future__ import annotations

import math
import os
import re
import random
from os.path import join
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from einops import rearrange
from torch import Tensor
from torch.utils import data
from torch.utils.data import random_split
from omegaconf import OmegaConf

from src.datamodules.abstract_datamodule import BaseDataModule
from src.datamodules.datasets.mdtrajectory import MDTrajectoryDataset, FullMDTrajectoryDataset
from src.datamodules.torch_datasets import MyTensorDataset
from src.utilities.utils import (
    get_logger,
    raise_error_if_invalid_type,
    raise_error_if_invalid_value,
    raise_if_invalid_shape,
)
from src.mol_frame.mychemical import initialize_chemdata

log = get_logger(__name__)


def prepare_batch(batch):
    padding_values = {
        "pro_aatype": 82, 'sm_aatype': 82, 'pro_mask': 0.0,
        'sm_mask': 0.0, "pro_esm": 0.0, "sm_esm": 0.0,
        "pro_T": 0.0, "pro_IR": 0.0, "sm_T": 0.0, "sm_IR": 0.0,
    }
    out = {}
    for key in padding_values.keys():
        out[key] = torch.nn.utils.rnn.pad_sequence([torch.tensor(
            x[key]) for x in batch], batch_first=True, padding_value=padding_values[key]).transpose(1, 2)
        if out[key].dtype == torch.int64 or out[key].dtype == torch.int16:
            out[key] = out[key].to(torch.int32)
        elif out[key].dtype == torch.float32 or out[key].dtype == torch.float64:
            out[key] = out[key].to(torch.float16)
    if "meta_data" in batch[0]:
        out["meta_data"] = [x["meta_data"] for x in batch]
    if "time" in batch[0]:
        out["time"] = torch.tensor([x["time"]
                                   for x in batch], dtype=torch.long)
    return out


class MolecularDynamicsSimulationDataModule(BaseDataModule):
    def __init__(
            self,
            data_dir: str,
            chem_config_dir: str,
            indices_file: str,
            window: int = 1,
            horizon: int = 1,
            valid_size: int = 10,
            test_size: int = 20,
            predict_size: int = 10,
            prediction_horizon=None,  # None means use horizon
            multi_horizon: bool = False,
            num_trajectories=None,  # None means all trajectories for training
            **kwargs,
    ):
        raise_error_if_invalid_type(
            data_dir, possible_types=[str], name="data_dir")
        raise_error_if_invalid_type(chem_config_dir, possible_types=[
                                    str], name="chem_config_dir")
        super().__init__(data_dir=data_dir, **kwargs)
        self.save_hyperparameters()
        self.base_dir = os.path.expanduser(data_dir)
        self.chem_config_dir = os.path.expanduser(chem_config_dir)
        self.indices_file = os.path.join(self.base_dir, indices_file)
        self.window = window
        self.horizon = horizon

        # to make sure that the test dataloader returns a single trajectory
        self.test_batch_size = 1
        assert window == 1, "window > 1 is not supported yet for this data module."

        # Check if data directory exists
        assert (
            os.path.isfile(self.chem_config_dir)
        ), f"Could not find data directory {chem_config_dir}. Is the data directory correct?. Please specify the data directory using the ``datamodule.chem_config_dir`` option."
        assert (
            os.path.isdir(self.base_dir)
        ), f"Could not find data directory {self.base_dir}. Is the data directory correct?. Please specify the data directory using the ``datamodule.data_dir`` option."
        assert (
            os.path.isfile(self.indices_file)
        ), f"Could not find indices_file {indices_file} in {self.base_dir}. Did you download the data?"
        log.info(f"Using data directory: {self.base_dir}")

        # initialize_chemdata
        config = OmegaConf.load(os.path.expanduser(self.chem_config_dir))
        initialize_chemdata(config.chem_params)

        # load indices file
        indices_dict = np.load(self.indices_file, allow_pickle=True).item()
        systems, indices_dict = self.find_unique_system(indices_dict)
        indices_dict = self.add_time(indices_dict)

        # append time information
        new_indices_dict = {sys: [] for sys in systems}
        for key, value in indices_dict.items():
            if value['time'] is not None:
                new_indices_dict[value['system']].append(value)

        # sort
        for key, value in new_indices_dict.items():
            new_indices_dict[key] = sorted(value, key=lambda x: x['time'])

        self.indices_dict = new_indices_dict
        self.n_trajectories = len(self.indices_dict)
        self.n_frames = {key: len(value)
                         for key, value in self.indices_dict.items()}

        for key, value in self.indices_dict.items():
            self.indices_dict[key] = [value[i:i + self.window + self.horizon]
                                      for i in range(self.n_frames[key] - self.window - self.horizon + 1)]

        # [[{},{},...,{}],[{},{},...,{}],...,[{},{},...,{}]]
        self.indices_merged = [item for sublist in self.indices_dict.values() for item in sublist]  
        total_size = len(self.indices_merged)
        train_size = total_size - valid_size - test_size - predict_size
        self._indices_train, self._indices_valid, self._indices_test, self._indices_predict = self.random_split_by_lengths(
            self.indices_merged, [train_size,
                                  valid_size, test_size, predict_size]
        )

    def random_split_by_lengths(self, input_list, lengths):
        """
        Randomly split a list into sublists of specified length

        Args:
            input_list: List to be partitioned
            lengths: A list of the lengths of each sublist
        Returns: 
            sublists: Lists containing sublists
        """
        if sum(lengths) != len(input_list):
            raise ValueError(
                "The sum of the sublist lengths must equal the length of the input list")

        random.shuffle(input_list)

        sublists = []
        start = 0

        for length in lengths:
            end = start + length
            sublists.append(input_list[start:end])
            start = end

        return sublists

    def find_unique_system(self, indices_dict: dict):
        """find unique systems given the indices file"""
        systems = set()
        for key, value in indices_dict.items():
            name = value['Npy-Pro Files'].lstrip('npy_pro/').rstrip('.npy')
            systems.add(name)
            indices_dict[key]['system'] = name
        return systems, indices_dict

    def add_time(self, indices_dict: dict):
        """
        add the time information given the indices file,
        e.g. for 'szs_MD/6-mv-lustre1-3-Sep.15-L-7alv-003-Sep-ph7.4.0ps-pro.pdb.0ps-pro.pdb'
        will add {'time': 0}
        """
        for key, value in indices_dict.items():
            pattern = r'(\d+)ps'
            match = re.search(pattern, key)
            if match:
                time_in_ps = int(match.group(1))
                indices_dict[key]['time'] = time_in_ps
            else:
                indices_dict[key]['time'] = None
        return indices_dict

    def get_horizon(self, split: str):
        if split in ["predict", "test"]:
            return getattr(self.hparams, "prediction_horizon") or getattr(self.hparams, "horizon")
        else:
            return getattr(self.hparams, "horizon")

    def setup(self, stage: Optional[str] = None):
        """Load data. Set internal variables: self._data_train, self._data_val, self._data_test."""
        assert stage in ["fit", "validate", "test",
                         "predict", None], f"Invalid stage {stage}"
        print(
            f"Setting up MolecularDynamicsSimulationDataModule for stage {stage}...")
        ds_train = MDTrajectoryDataset(
            self.base_dir, self._indices_train, self.window, self.horizon
        ) if stage in ["fit", None] else None
        ds_val = FullMDTrajectoryDataset(
            self.base_dir, self._indices_valid, self.window, self.horizon
        ) if stage in ["fit", "validate", None] else None
        ds_test = FullMDTrajectoryDataset(
            self.base_dir, self._indices_test, self.window, self.horizon
        ) if stage in ["test", None] else None
        ds_predict = FullMDTrajectoryDataset(
            self.base_dir, self._indices_predict, self.window, self.horizon
        ) if stage == "predict" else None
        ds_splits = {"train": ds_train, "val": ds_val,
                     "test": ds_test, "predict": ds_predict}
        for split, split_ds in ds_splits.items():
            if split_ds is None:
                continue
            setattr(self, f"_data_{split}", split_ds)
            assert getattr(
                self, f"_data_{split}") is not None, f"Could not create {split} dataset"

        # Print sizes of the datasets (how many examples)
        if stage is not None:
            self.print_data_sizes(stage)

    def _shared_dataloader_kwargs(self) -> dict:
        return dict(
            num_workers=self.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            collate_fn=prepare_batch
        )
