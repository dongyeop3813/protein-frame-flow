from typing import Any
import torch
import time
import os
import random
import wandb
import numpy as np
import pandas as pd
import logging
import torch.distributed as dist
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import grad_norm
from analysis import metrics
from analysis import utils as au
from models.meanflow_model import MeanFlowModel
from models import utils as mu
from data.interpolant import SplitMeanFlowInterpolant
from data import utils as du
from data import all_atom
from data import so3_utils
from data import residue_constants
from experiments import utils as eu
from pytorch_lightning.loggers.wandb import WandbLogger


class SplitMeanFlowModule(LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self._print_logger = logging.getLogger(__name__)
        self._exp_cfg = cfg.experiment
        self._model_cfg = cfg.model
        self._data_cfg = cfg.data
        self._interpolant_cfg = cfg.interpolant

        # Set-up vector field prediction model
        self.model = MeanFlowModel(cfg.model)

        # Set-up interpolant
        self.interpolant = SplitMeanFlowInterpolant(cfg.interpolant)

        self.model.set_interpolant(self.interpolant)

        self.validation_epoch_metrics = []
        self.validation_epoch_one_step_metrics = []
        self.validation_epoch_samples = []
        self.validation_epoch_one_step_samples = []
        self.save_hyperparameters()

        self._checkpoint_dir = None
        self._inference_dir = None

    @property
    def checkpoint_dir(self):
        if self._checkpoint_dir is None:
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    checkpoint_dir = [self._exp_cfg.checkpointer.dirpath]
                else:
                    checkpoint_dir = [None]
                dist.broadcast_object_list(checkpoint_dir, src=0)
                checkpoint_dir = checkpoint_dir[0]
            else:
                checkpoint_dir = self._exp_cfg.checkpointer.dirpath
            self._checkpoint_dir = checkpoint_dir
            os.makedirs(self._checkpoint_dir, exist_ok=True)
        return self._checkpoint_dir

    @property
    def inference_dir(self):
        if self._inference_dir is None:
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    inference_dir = [self._exp_cfg.inference_dir]
                else:
                    inference_dir = [None]
                dist.broadcast_object_list(inference_dir, src=0)
                inference_dir = inference_dir[0]
            else:
                inference_dir = self._exp_cfg.inference_dir
            self._inference_dir = inference_dir
            os.makedirs(self._inference_dir, exist_ok=True)
        return self._inference_dir

    def on_train_start(self):
        self._epoch_start_time = time.time()

    def on_train_epoch_end(self):
        epoch_time = (time.time() - self._epoch_start_time) / 60.0
        self.log(
            "train/epoch_time_minutes",
            epoch_time,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self._epoch_start_time = time.time()

    def flow_matching_loss(self, trans_t, rot_t, trans_1, rot_1, t, loss_mask, feats):
        # Conditional velocity fields
        # Here, v_rot is a rotation vector.
        loss_denom = torch.sum(loss_mask, dim=-1) * 3

        one_minus_t = torch.clamp(1 - t, min=1e-4)[..., None]
        v_trans = (trans_1 - trans_t) / one_minus_t
        v_rot = so3_utils.calc_rot_vf(rot_t, rot_1) / one_minus_t
        if torch.any(torch.isnan(v_rot)):
            raise ValueError("NaN encountered in v_rot")

        # Flow matching loss.
        u_trans_t, u_rot_t = self.model.avg_vel(trans_t, rot_t, t, t, feats)

        # Flow matching on translation.
        trans_loss = torch.sum(
            ((u_trans_t - v_trans) * loss_mask[..., None]) ** 2, dim=(-1, -2)
        )

        # Flow matching on rotation.
        rot_loss = torch.sum(
            ((u_rot_t - v_rot) * loss_mask[..., None]) ** 2, dim=(-1, -2)
        )

        # Normalize by the number of residues.
        trans_loss /= loss_denom
        rot_loss /= loss_denom

        return trans_loss, rot_loss

    def semigroup_loss(self, trans_t, rot_t, trans_r, rot_r, t, r, loss_mask, feats):
        loss_denom = torch.sum(loss_mask, dim=-1) * 3

        r_minus_t = torch.clamp(r - t, min=1e-4)[..., None]
        u_trans_tgt = (trans_r - trans_t) / r_minus_t
        u_rot_tgt = so3_utils.calc_rot_vf(rot_t, rot_r) / r_minus_t

        u_trans, u_rot = self.model.avg_vel(trans_t, rot_t, t, r, feats)

        # Semigroup loss on translation.
        trans_loss = torch.sum(
            ((u_trans - u_trans_tgt) * loss_mask[..., None]) ** 2, dim=(-1, -2)
        )

        # Semigroup loss on rotation.
        rot_loss = torch.sum(
            ((u_rot - u_rot_tgt) * loss_mask[..., None]) ** 2, dim=(-1, -2)
        )

        # Normalize by the number of residues.
        trans_loss /= loss_denom
        rot_loss /= loss_denom

        return trans_loss, rot_loss

    def model_step(self, noisy_batch: Any):
        training_cfg = self._exp_cfg.training
        loss_mask = noisy_batch["res_mask"] * noisy_batch["diffuse_mask"]
        if torch.any(torch.sum(loss_mask, dim=-1) < 1):
            raise ValueError("Empty batch encountered")

        # Extract values from noisy batch.
        trans_1 = noisy_batch["trans_1"]
        trans_t = noisy_batch["trans_t"]
        rot_1 = noisy_batch["rotmats_1"]
        rot_t = noisy_batch["rotmats_t"]
        xt = (trans_t, rot_t)
        x1 = (trans_1, rot_1)

        assert (noisy_batch["r3_t"] == noisy_batch["so3_t"]).all()
        t = noisy_batch["so3_t"]
        s = noisy_batch["s"]
        r = noisy_batch["r"]

        feat = noisy_batch

        trans_loss, rot_loss = self.flow_matching_loss(*xt, *x1, t, loss_mask, feat)
        weighted_trans_loss = trans_loss * training_cfg.translation_loss_weight
        weighted_rot_loss = rot_loss * training_cfg.rotation_loss_weight
        fm_loss = weighted_trans_loss + weighted_rot_loss
        if torch.any(torch.isnan(fm_loss)):
            raise ValueError("NaN loss encountered")

        # Algebraic consistency.
        with torch.no_grad():
            hat_xs = self.model.forward_flow(*xt, t, s, feat)
            hat_xr = self.model.forward_flow(*hat_xs, s, r, feat)

        sg_trans_loss, sg_rot_loss = self.semigroup_loss(
            *xt, *hat_xr, t, r, loss_mask, feat
        )
        weighted_sg_trans_loss = sg_trans_loss * training_cfg.translation_loss_weight
        weighted_sg_rot_loss = sg_rot_loss * training_cfg.rotation_loss_weight
        semigroup_loss = weighted_sg_trans_loss + weighted_sg_rot_loss
        if torch.any(torch.isnan(semigroup_loss)):
            raise ValueError("NaN loss encountered")

        loss = (
            training_cfg.flow_matching_loss_weight * fm_loss
            + training_cfg.semigroup_loss_weight * semigroup_loss
        )
        return {
            "fm_trans_loss": trans_loss,
            "fm_rot_loss": rot_loss,
            "weighted_fm_trans_loss": weighted_trans_loss,
            "weighted_fm_rot_loss": weighted_rot_loss,
            "sg_trans_loss": sg_trans_loss,
            "sg_rot_loss": sg_rot_loss,
            "weighted_sg_trans_loss": weighted_sg_trans_loss,
            "weighted_sg_rot_loss": weighted_sg_rot_loss,
            "semigroup_loss": semigroup_loss,
            "split_meanflow_loss": loss,
        }

    def validation_step(self, batch: Any, batch_idx: int):
        res_mask = batch["res_mask"]
        self.interpolant.set_device(res_mask.device)
        num_batch, num_res = res_mask.shape
        diffuse_mask = batch["diffuse_mask"]
        csv_idx = batch["csv_idx"]

        ####### Original validation code (with integration) #######
        atom37_traj, _, _ = self.interpolant.sample(
            num_batch,
            num_res,
            self.model,
            trans_1=batch["trans_1"],
            rotmats_1=batch["rotmats_1"],
            diffuse_mask=diffuse_mask,
            chain_idx=batch["chain_idx"],
            res_idx=batch["res_idx"],
        )
        samples = atom37_traj[-1].numpy()
        batch_metrics = []
        for i in range(num_batch):
            sample_dir = os.path.join(
                self.checkpoint_dir,
                f"sample_{csv_idx[i].item()}_idx_{batch_idx}_len_{num_res}",
            )
            os.makedirs(sample_dir, exist_ok=True)

            # Write out sample to PDB file
            final_pos = samples[i]
            saved_path = au.write_prot_to_pdb(
                final_pos, os.path.join(sample_dir, "sample.pdb"), no_indexing=True
            )
            if isinstance(self.logger, WandbLogger):
                self.validation_epoch_samples.append(
                    [saved_path, self.global_step, wandb.Molecule(saved_path)]
                )

            mdtraj_metrics = metrics.calc_mdtraj_metrics(saved_path)
            ca_idx = residue_constants.atom_order["CA"]
            ca_ca_metrics = metrics.calc_ca_ca_metrics(final_pos[:, ca_idx])
            batch_metrics.append((mdtraj_metrics | ca_ca_metrics))

        batch_metrics = pd.DataFrame(batch_metrics)
        self.validation_epoch_metrics.append(batch_metrics)

        ####### One step validation code #######
        trans_0, rotmats_0 = self.interpolant.init_prior(num_batch, num_res)
        atom37 = self.interpolant.forward_flow(
            num_batch,
            num_res,
            self.model,
            trans_t=trans_0,
            rotmats_t=rotmats_0,
            t=0.0,
            r=1.0,
        )
        samples = atom37.numpy()
        batch_metrics = []
        for i in range(num_batch):
            sample_dir = os.path.join(
                self.checkpoint_dir,
                f"one_step_sample_{csv_idx[i].item()}_idx_{batch_idx}_len_{num_res}",
            )
            os.makedirs(sample_dir, exist_ok=True)

            # Write out sample to PDB file
            final_pos = samples[i]
            saved_path = au.write_prot_to_pdb(
                final_pos, os.path.join(sample_dir, "sample.pdb"), no_indexing=True
            )
            if isinstance(self.logger, WandbLogger):
                self.validation_epoch_one_step_samples.append(
                    [saved_path, self.global_step, wandb.Molecule(saved_path)]
                )

            mdtraj_metrics = metrics.calc_mdtraj_metrics(saved_path)
            ca_idx = residue_constants.atom_order["CA"]
            ca_ca_metrics = metrics.calc_ca_ca_metrics(final_pos[:, ca_idx])
            batch_metrics.append((mdtraj_metrics | ca_ca_metrics))

        batch_metrics = pd.DataFrame(batch_metrics)
        self.validation_epoch_one_step_metrics.append(batch_metrics)

    def on_validation_epoch_end(self):
        if len(self.validation_epoch_samples) > 0:
            self.logger.log_table(
                key="valid/samples",
                columns=["sample_path", "global_step", "Protein"],
                data=self.validation_epoch_samples,
            )
            self.validation_epoch_samples.clear()
        val_epoch_metrics = pd.concat(self.validation_epoch_metrics)
        for metric_name, metric_val in val_epoch_metrics.mean().to_dict().items():
            self._log_scalar(
                f"valid/{metric_name}",
                metric_val,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=len(val_epoch_metrics),
            )
        self.validation_epoch_metrics.clear()

        if len(self.validation_epoch_one_step_samples) > 0:
            self.logger.log_table(
                key="valid/one_step_samples",
                columns=["sample_path", "global_step", "Protein"],
                data=self.validation_epoch_one_step_samples,
            )
            self.validation_epoch_one_step_samples.clear()
        val_epoch_metrics = pd.concat(self.validation_epoch_one_step_metrics)
        for metric_name, metric_val in val_epoch_metrics.mean().to_dict().items():
            self._log_scalar(
                f"valid/one_step_{metric_name}",
                metric_val,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=len(val_epoch_metrics),
            )
        self.validation_epoch_one_step_metrics.clear()

    def _log_scalar(
        self,
        key,
        value,
        on_step=True,
        on_epoch=False,
        prog_bar=True,
        batch_size=None,
        sync_dist=False,
        rank_zero_only=True,
    ):
        if sync_dist and rank_zero_only:
            raise ValueError("Unable to sync dist when rank_zero_only=True")
        self.log(
            key,
            value,
            on_step=on_step,
            on_epoch=on_epoch,
            prog_bar=prog_bar,
            batch_size=batch_size,
            sync_dist=sync_dist,
            rank_zero_only=rank_zero_only,
        )

    def training_step(self, batch: Any, stage: int):
        step_start_time = time.time()
        self.interpolant.set_device(batch["res_mask"].device)
        noisy_batch = self.interpolant.corrupt_batch(batch)

        # No self-conditioning, no auxiliary losses.
        batch_losses = self.model_step(noisy_batch)
        num_batch = batch_losses["fm_trans_loss"].shape[0]

        total_losses = {k: torch.mean(v) for k, v in batch_losses.items()}
        for k, v in total_losses.items():
            self._log_scalar(f"train/{k}", v, prog_bar=False, batch_size=num_batch)

        # Losses to track. Stratified across t.
        for loss_name, batch_loss in batch_losses.items():
            batch_t = noisy_batch["so3_t"]

            # Bin the loss by timestep.
            stratified_losses = mu.t_stratified_loss(
                batch_t, batch_loss, loss_name=loss_name
            )

            # Log the stratified losses.
            for k, v in stratified_losses.items():
                self._log_scalar(
                    f"train/stratified_loss/{k}",
                    v,
                    prog_bar=False,
                    batch_size=num_batch,
                )

        # Training throughput
        self._log_scalar(
            "train/length",
            batch["res_mask"].shape[1],
            prog_bar=False,
            batch_size=num_batch,
        )
        self._log_scalar("train/batch_size", num_batch, prog_bar=False)
        step_time = time.time() - step_start_time
        self._log_scalar("train/examples_per_second", num_batch / step_time)

        # This is the final training objective.
        train_loss = total_losses["split_meanflow_loss"]
        self._log_scalar("train/loss", train_loss, batch_size=num_batch)
        return train_loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            params=self.model.parameters(), **self._exp_cfg.optimizer
        )

    def on_before_optimizer_step(self, optimizer):
        # Compute grad norms and rename keys for cleaner logging
        raw_norms = grad_norm(self.model, norm_type=2, group_separator="/")
        if not raw_norms:
            return
        cleaned_norms = {}
        for key, value in raw_norms.items():
            if key.endswith("_total"):
                cleaned_norms["grad_norm/total"] = value
            else:
                # Original key example: "grad_2.0_norm/<param_path>"
                suffix = key.split("/", 1)[1] if "/" in key else key
                cleaned_norms[f"grad_norm/layers/{suffix}"] = value
        self.log_dict(cleaned_norms)

    def predict_step(self, batch, batch_idx):
        del batch_idx  # Unused
        device = f"cuda:{torch.cuda.current_device()}"
        interpolant = SplitMeanFlowInterpolant(self._infer_cfg.interpolant)
        interpolant.set_device(device)

        sample_ids = batch["sample_id"].squeeze().tolist()
        sample_ids = [sample_ids] if isinstance(sample_ids, int) else sample_ids
        num_batch = len(sample_ids)

        if "diffuse_mask" in batch:  # motif-scaffolding
            target = batch["target"][0]
            trans_1 = batch["trans_1"]
            rotmats_1 = batch["rotmats_1"]
            diffuse_mask = batch["diffuse_mask"]
            true_bb_pos = all_atom.atom37_from_trans_rot(
                trans_1, rotmats_1, 1 - diffuse_mask
            )
            true_bb_pos = true_bb_pos[..., :3, :].reshape(-1, 3).cpu().numpy()
            _, sample_length, _ = trans_1.shape
            sample_dirs = [
                os.path.join(self.inference_dir, target, f"sample_{str(sample_id)}")
                for sample_id in sample_ids
            ]
        else:  # unconditional
            sample_length = batch["num_res"].item()
            true_bb_pos = None
            sample_dirs = [
                os.path.join(
                    self.inference_dir,
                    f"length_{sample_length}",
                    f"sample_{str(sample_id)}",
                )
                for sample_id in sample_ids
            ]
            trans_1 = rotmats_1 = diffuse_mask = None
            diffuse_mask = torch.ones(1, sample_length, device=device)

        # Sample batch
        atom37_traj, model_traj, _ = interpolant.sample(
            num_batch,
            sample_length,
            self.model,
            trans_1=trans_1,
            rotmats_1=rotmats_1,
            diffuse_mask=diffuse_mask,
        )

        bb_trajs = du.to_numpy(torch.stack(atom37_traj, dim=0).transpose(0, 1))
        for i in range(num_batch):
            sample_dir = sample_dirs[i]
            bb_traj = bb_trajs[i]
            os.makedirs(sample_dir, exist_ok=True)
            if "aatype" in batch:
                aatype = du.to_numpy(batch["aatype"].long())[0]
            else:
                aatype = np.zeros(sample_length, dtype=int)
            _ = eu.save_traj(
                bb_traj[-1],
                bb_traj,
                np.flip(du.to_numpy(torch.concat(model_traj, dim=0)), axis=0),
                du.to_numpy(diffuse_mask)[0],
                output_dir=sample_dir,
                aatype=aatype,
            )


def weighted_loss(loss, p=0.0):
    """
    Adaptive weighting of losses.

    Args:
        loss: The loss to weight.
        p: The power to weight the loss by.

    Returns:
        The weighted loss.
    """
    weight = 1 / ((loss + 1e-3) ** p)
    return weight.detach() * loss
