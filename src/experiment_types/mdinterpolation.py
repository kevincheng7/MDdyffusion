import inspect
import os
from typing import Any, List, Dict, Optional

import numpy as np
import hydra
import torch
import torchmetrics
from einops import rearrange
from torch import Tensor
from scipy.spatial.transform import Rotation as R
from openbabel import openbabel, pybel

from src.models._base_model import BaseModel
from src.experiment_types._base_experiment import BaseExperiment
from src.mol_frame.mychemical import ChemicalData


class RootMeanSquareDeviation(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("sum_squared_distances",
                       default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_elements", default=torch.tensor(0),
                       dist_reduce_fx="sum")

    def convert_to_CANC(self, T, IR):
        CA = T  # (b, n, 3)
        N_ref = torch.tensor([1.45597958, 0.0, 0.0], device=T.device)  # (3, )
        C_ref = torch.tensor(
            [-0.533655602, 1.42752619, 0.0], device=T.device)  # (3, )
        N = torch.matmul(IR.transpose(-1, -2), N_ref) + CA
        C = torch.matmul(IR.transpose(-1, -2), C_ref) + CA
        return CA, N, C

    def update(self, predict_T, target_T, num_residue, num_ligand_atom, padding_mask=None):
        """
        Args:
            predict_T (`tensor`, `tensor`): ([batch, N_frame, 3], [batch, N_frame, 3, 3])
            target_T (`tensor`, `tensor`): ([batch, N_frame, 3], [batch, N_frame, 3, 3])
            num_residue (`int`)
            num_ligand_atom (`int`)
            padding_mask (`tensor`, optional): padding mask. size: [batch, N_frame]. Defaults to None.
        """
        predict_Trans, predict_Rot = predict_T
        target_Trans, target_Rot = target_T
        pred_CA, pred_N, pred_C = self.convert_to_CANC(
            predict_Trans[:, :num_residue, :], predict_Rot[:, :num_residue, :, :])
        pred_LIG = predict_Trans[:, num_residue:, :]

        target_CA, target_N, target_C = self.convert_to_CANC(
            target_Trans[:, :num_residue, :], target_Rot[:, :num_residue, :, :])
        target_LIG = target_Trans[:, num_residue:, :]

        pred = torch.cat([pred_CA, pred_C, pred_N, pred_LIG], dim=1)
        target = torch.cat([target_CA, target_C, target_N, target_LIG], dim=1)
        squared_distance = torch.linalg.vector_norm(
            pred - target, dim=-1) ** 2  # (batch, N_atom)
        if padding_mask is not None:
            padding_mask = torch.cat(
                [padding_mask[:, :num_residue].repeat(1, 3), padding_mask[:, num_residue:]], dim=1
            )
            squared_distance = squared_distance * padding_mask

        self.sum_squared_distances = self.sum_squared_distances + \
            torch.sum(squared_distance)
        self.num_elements = self.num_elements + torch.numel(squared_distance)

    def compute(self):
        return torch.sqrt(self.sum_squared_distances / self.num_elements)


class MDInterpolationExperiment(BaseExperiment):
    r"""Base class for MD interpolation experiments."""

    def __init__(self, save_dir: str = "", **kwargs):
        super().__init__(**kwargs)
        # The following saves all the args that are passed to the constructor to self.hparams
        #   e.g. access them with self.hparams.hidden_dims
        self.save_hyperparameters(ignore=["model"])
        assert self.horizon >= 2, "horizon must be >=2 for interpolation experiments"

    def instantiate_model(self, *args, **kwargs) -> BaseModel:
        r"""Instantiate the model, e.g. by calling the constructor of the class :class:`BaseModel` or a subclass thereof."""

        in_channels = self.actual_num_input_channels(self.dims["input"])  # 0
        out_channels = self.actual_num_output_channels(
            self.dims["output"])  # 0
        cond_channels = self.num_conditional_channels
        kwargs["datamodule_config"] = self.datamodule_config

        model = hydra.utils.instantiate(
            self.model_config,
            num_input_channels=in_channels,
            num_output_channels=out_channels,
            num_conditional_channels=cond_channels,
            spatial_shape=self.dims["spatial"],
            _recursive_=False,
            **kwargs,
        )
        self.log_text.info(f"Instantiated model: {model.__class__.__name__}")
        if self.is_diffusion_model:
            model = hydra.utils.instantiate(
                self.diffusion_config, model=model, _recursive_=False, **kwargs)
            self.log_text.info(
                f"Instantiated diffusion model: {model.__class__.__name__}, with"
                f" #diffusion steps={model.num_timesteps}"
            )
        return model

    @property
    def horizon_range(self) -> List[int]:
        # h = horizon
        # We use timesteps  w-l+1, ..., w-1, w+h to predict timesteps w, ..., w+h-1
        # interpolate between step t=0 and t=horizon
        return list(np.arange(1, self.horizon))

    @property
    def true_horizon(self) -> int:
        return self.horizon

    @property
    def horizon_name(self) -> str:
        s = f"{self.true_horizon}h"
        return s

    @property
    def short_description(self) -> str:
        name = super().short_description
        name += f" (h={self.horizon_name})"
        return name

    @property
    def WANDB_LAST_SEP(self) -> str:
        return "/ipol/"

    # --------------------------------- Metrics
    def get_metrics(self, split: str, split_name: str, **kwargs) -> torch.nn.ModuleDict:
        metrics = {
            f"{split_name}/{self.horizon_name}_avg{self.WANDB_LAST_SEP}rmsd": RootMeanSquareDeviation()
        }
        for h in self.horizon_range:
            metrics[f"{split_name}/t{h}{self.WANDB_LAST_SEP}rmsd"] = RootMeanSquareDeviation()
        return torch.nn.ModuleDict(metrics)

    @property
    def default_monitor_metric(self) -> str:
        return f"val/{self.horizon_name}_avg{self.WANDB_LAST_SEP}rmsd"

    def convert_to_CANC(self, T, IR):
        CA = T  # (b, n, 3)
        N_ref = torch.tensor([1.45597958, 0.0, 0.0], device=T.device)  # (3, )
        C_ref = torch.tensor(
            [-0.533655602, 1.42752619, 0.0], device=T.device)  # (3, )
        N = torch.matmul(IR.transpose(-1, -2), N_ref) + CA
        C = torch.matmul(IR.transpose(-1, -2), C_ref) + CA
        return CA, N, C

    def save_to_pdb(
            self,
            pro_aatype: Tensor,
            sm_aatype: Tensor,
            T: Tensor,
            IR: Tensor,
            t_step: int,
            meta_data: List,
            suffix: str = ""
    ):
        """
        save a batch of structure to pdb file.
        Args:
            pro_aatype (tensor): protein type, of shape (b, w+h, n_residue).
            sm_aatype (tensor): ligand type, of shape (b, w+h, n_ligand_atom).
            T (tensor): translation, of shape (b, n, 3).
            IR (tensor): rotation matrix, of shape (b, n, 3, 3).
            t_step (int): the current step of interpolation.
            meta_data (list): used to infer the file name, of length b:
                [
                    [{},{},...,{}],
                    ...
                    [{},{},...,{}]
                ].
            suffix (str): suffix of the file name.
        """
        save_dir = getattr(self.hparams, "save_dir")
        assert save_dir is not None, "please specify save_dir during initialization"

        b = T.size(dim=0)
        num_residue = pro_aatype.size(dim=-1)
        num_ligand_atom = sm_aatype.size(dim=-1)

        CAs, Ns, Cs = self.convert_to_CANC(
            T[:, :num_residue, :], IR[:, :num_residue, :, :])
        LIGs = T[:, num_residue:, :]

        for b_idx, (CA, N, C, LIG) in enumerate(zip(CAs, Ns, Cs, LIGs)):
            system = meta_data[b_idx][self.window + t_step - 1]["system"]
            time = meta_data[b_idx][self.window + t_step - 1]["time"]
            # (n_residue, )
            pro_type = pro_aatype[b_idx][self.window + t_step - 1]
            # (n_ligand_atom, )
            sm_type = sm_aatype[b_idx][self.window + t_step - 1]

            file_name = f"{system}_{time}_{suffix}.pdb"
            save_dir = os.path.join(save_dir, f"{system}")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            file_name = os.path.join(save_dir, file_name)

            aa2index = [
                'ALA', 'ARG', 'ASN', 'ASP', 'CYS',
                'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                'LEU', 'LYS', 'MET', 'PHE', 'PRO',
                'SER', 'THR', 'TRP', 'TYR', 'VAL'
            ]
            residues = [aa2index[pro_type[i]] for i in range(num_residue)]
            ligand_atomtype = [ChemicalData().num2aa[sm_type[i]]
                               for i in range(num_ligand_atom)]
            ligand_atomnum = [ChemicalData().atomtype2atomnum[ligand_atomtype[i]]
                              for i in range(num_ligand_atom)]

            line = "ATOM%7i  %s  %s A%4i    %8.3f%8.3f%8.3f  1.00  0.00           %s"
            lines = []
            for i in range(num_residue):
                lines.append(
                    line
                    % (3 * i + 1, "CA", residues[i], i + 1, CA[i][0], CA[i][1], CA[i][2], "C")
                )
                lines.append(
                    line
                    % (3 * i + 2, " C", residues[i], i + 1, C[i][0], C[i][1], C[i][2], "C")
                )
                lines.append(
                    line
                    % (3 * i + 3, " N", residues[i], i + 1, N[i][0], N[i][1], N[i][2], "N")
                )
            lines.append("TER")
            lines = "\n".join(lines)

            obConversion = openbabel.OBConversion()
            obConversion.SetInFormat("pdb")
            obmol_protein = openbabel.OBMol()
            obConversion.ReadString(obmol_protein, lines)

            obmol_ligand = openbabel.OBMol()
            for i in range(num_ligand_atom):
                atom = obmol_ligand.NewAtom()
                atom.SetVector(LIG[i, 0].item(),
                               LIG[i, 1].item(), LIG[i, 2].item())
                atom.SetAtomicNum(ligand_atomnum[i])

            mol_protein = pybel.Molecule(obmol_protein)
            mol_ligand = pybel.Molecule(obmol_ligand)

            file = pybel.Outputfile('pdb', file_name, overwrite=True)
            file.write(mol_protein)
            file.write(mol_ligand)
            file.close()

    @torch.no_grad()
    def _evaluation_step(
        self,
        batch: Any,
        batch_idx: int,
        split: str,
        dataloader_idx=None,
        return_only_preds_and_targets: bool = False,
    ) -> Dict[str, Tensor]:
        """
        One step of evaluation (forward pass, potentially metrics computation, logging, and return of results)
        Returns:
            results_dict: Dict[str, Tensor], where for each semantically different result, a separate prefix key is used
                Then, for each prefix key <p>, results_dict must contain <p>_preds and <p>_targets.
        """
        log_dict = dict()
        compute_metrics = split != "predict"
        save_to_pdb = split in ["test", "predict"]
        split_metrics = getattr(
            self, f"{split}_metrics") if compute_metrics else None
        b = batch['pro_aatype'].shape[0]
        num_residue = batch['pro_aatype'].size(dim=-1)
        num_ligand_atom = batch['sm_aatype'].size(dim=-1)

        effective_batch_size = b * getattr(self.hparams, "num_predictions")

        return_dict = dict()
        avg_rmsd_key = f"{split}/{self.horizon_name}_avg{self.WANDB_LAST_SEP}rmsd"
        avg_rmsd_tracker = split_metrics[avg_rmsd_key] if split_metrics is not None else None

        inputs, meta_data = self.get_evaluation_inputs(batch, split=split)

        for t_step in self.horizon_range:
            targets = dict()
            for key, value in batch.items():
                targets[key] = value[torch.arange(
                    b), self.window + t_step - 1, ...]
            targets_T = torch.cat([targets['pro_T'], targets['sm_T']], dim=1)
            targets_IR = torch.cat(
                [targets['pro_IR'], targets['sm_IR']], dim=1)
            time = torch.full((effective_batch_size,), t_step,
                              device=self.device, dtype=torch.long)

            results = self.predict(inputs, time=time)
            results["targets_T"] = targets_T  # (b, n, 3)
            results["targets_IR"] = targets_IR  # (b, n, 3, 3)
            preds_T = results["preds_T"]  # (n_ensemble, b, n, 3)
            preds_IR = results["preds_IR"]  # (n_ensemble, b, n, 3, 3)
            preds_mask = results["mask"]  # (n_ensemble, b, n)
            results = {f"t{t_step}_{k}": v for k, v in results.items()}

            if return_only_preds_and_targets:
                return_dict[f"t{t_step}_preds_T"] = preds_T
                return_dict[f"t{t_step}_preds_IR"] = preds_IR
                return_dict[f"t{t_step}_targets_T"] = targets_T
                return_dict[f"t{t_step}_targets_IR"] = targets_IR
            else:
                return_dict = {**return_dict, **results}

            if self.use_ensemble_predictions(split):
                # average over ensemble
                preds_T = preds_T.mean(dim=0)
                preds_IR_list = []
                for b_idx in range(b):
                    tmp_IR = np.stack(  # (n, 3, 3)
                        [R.from_matrix(np.array(preds_IR[:, b_idx, n, ...].cpu())).mean().as_matrix(
                        ) for n in range(num_residue + num_ligand_atom)]
                    )
                    preds_IR_list.append(tmp_IR)
                preds_IR = np.stack(preds_IR_list)
                preds_IR = torch.tensor(
                    preds_IR, device=preds_T.device, dtype=preds_T.dtype)
                # use the first mask since they are the same
                preds_mask = preds_mask[0]

            if save_to_pdb:
                self.save_to_pdb(batch["pro_aatype"], batch["sm_aatype"],
                                 preds_T, preds_IR, t_step, meta_data, suffix="pred")
                self.save_to_pdb(batch["pro_aatype"], batch["sm_aatype"],
                                 targets_T, targets_IR, t_step, meta_data, suffix="target")

            if not compute_metrics:
                continue
            # Compute rmsd
            assert split_metrics is not None and avg_rmsd_tracker is not None
            metric_name = f"{split}/t{t_step}{self.WANDB_LAST_SEP}rmsd"
            metric = split_metrics[metric_name]
            metric((preds_T, preds_IR), (targets_T, targets_IR),
                   num_residue, num_ligand_atom, preds_mask)
            log_dict[metric_name] = metric

            # Add contribution to the average rmsd from this time step's rsmd
            avg_rmsd_tracker((preds_T, preds_IR), (targets_T, targets_IR),
                             num_residue, num_ligand_atom, preds_mask)

        if compute_metrics:
            log_kwargs = dict()
            log_kwargs["sync_dist"] = True  # for DDP training
            # Log the average MSE
            log_dict[avg_rmsd_key] = avg_rmsd_tracker
            self.log_dict(log_dict, on_step=False, on_epoch=True,
                          **log_kwargs)  # log metric objects

        return return_dict

    def predict(
        self, inputs: dict, num_predictions: Optional[int] = None, reshape_ensemble_dim: bool = True, **kwargs
    ) -> Dict[str, Tensor]:
        """
        Modified predict method, where inputs is a dictionary. Used in MDInterpolationExperiment.
        This should be the main method to use for making predictions/doing inference.

        Args:
            inputs (dict): Input data dictionary.
                This is the same tensor one would use in :func:`forward`.
            num_predictions (int, optional): Number of predictions to make. If None, use the default value.
            reshape_ensemble_dim (bool, optional): Whether to reshape the ensemble dimension into the first dimension.
            **kwargs: Additional keyword arguments

        Returns:
            Dict[str, Tensor]: The model predictions (in a post-processed format), i.e. a dictionary output_var -> output_var_prediction,
                where each output_var_prediction is a Tensor of shape :math:`(B, *)` in original-scale (e.g.
                in Kelvin for temperature), and non-negativity has been enforced for variables such as precipitation.
        """
        base_num_predictions = getattr(self.hparams, "num_predictions")
        self.hparams.num_predictions = num_predictions or base_num_predictions
        if (
            hasattr(self.model, "sample_loop")
            and "num_predictions" in inspect.signature(getattr(self.model, "sample_loop")).parameters
        ):
            kwargs["num_predictions"] = getattr(
                self.hparams, "num_predictions")

        (preds_T, preds_IR), mask = self.model.predict_forward(
            inputs, **kwargs)  # by default, just call the forward method
        results = {"preds_T": preds_T, "preds_IR": preds_IR, "mask": mask}

        self.hparams.num_predictions = base_num_predictions
        results = self.reshape_predictions(results, reshape_ensemble_dim)
        results = self.unpack_predictions(results)

        return results

    def get_evaluation_inputs(self, batch: dict, split: str, **kwargs):
        """Get the network inputs from the dynamics data.
        Since we are doing interpolation, this consists of the first window frames plus the last frame.
        """
        assert split in [
            "val", "test", "predict"], "split should be 'val', 'test' or 'predict', please assign the right split"
        meta_data = batch.pop('meta_data')
        batch = self.get_ensemble_inputs(
            batch, split, add_noise=False)  # type: ignore
        inputs = dict()
        for key, value in batch.items():
            assert value.shape[1] == self.window + \
                self.horizon, "dynamics must have shape (b * num_predictions, window + horizon, *)"
            # (b * num_predictions, window, *) at time 0
            past_steps = value[:, : self.window, ...]
            if self.window == 1:
                # (b * num_predictions, *) at time 0
                past_steps = past_steps.squeeze(1)
            # (b * num_predictions, *) at time t=window+horizon
            last_step = value[:, -1, ...]
            inputs[key] = [past_steps, last_step]
        return inputs, meta_data

    def get_training_inputs(self, batch: dict, split: str, **kwargs):
        """Get the network inputs from the dynamics data.
        Since we are doing interpolation, this consists of the first window frames plus the last frame.
        """
        assert split == "train", "split should be 'train', please assign the right split"
        t = batch.pop('time')
        meta_data = batch.pop('meta_data')
        inputs = dict()
        for key, value in batch.items():
            assert value.shape[1] == self.window + \
                2, "dynamics must have shape (b, window + 2, *)"
            # (b, window, *) at time 0
            past_steps = value[:, : self.window, ...]
            if self.window == 1:
                past_steps = past_steps.squeeze(1)  # (b, *) at time 0
            last_step = value[:, -1, ...]  # (b, *) at time t=window+horizon
            inputs[key] = [past_steps, last_step]
        return inputs, t, meta_data

    # --------------------------------- Training
    def get_loss(self, batch: Any):
        r"""Compute the loss for the given batch."""

        split = "train" if self.training else "val"
        b = batch['pro_aatype'].shape[0]
        inputs, t, meta_data = self.get_training_inputs(batch, split=split)

        targets = dict()
        for key, value in batch.items():
            targets[key] = value[torch.arange(b), self.window, ...]
        # We use timesteps  w-l+1, ..., w-1, w+h to predict timesteps w, ..., w+h-1
        # so t=0 corresponds to interpolating w, t=1 to w+1, ..., t=h-1 to w+h-1

        loss = self.model.get_loss(
            inputs=inputs, targets=targets, time=t,  # type: ignore
        )
        return loss
