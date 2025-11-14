import os
import time
import logging
import random

import torch
import torch.distributed as dist
import numpy as np
import pandas as pd

import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

from typing import Any

from pytorch_lightning import (
    seed_everything,
    LightningDataModule,
    LightningModule,
    Trainer,
)
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import grad_norm

from scipy.spatial.transform import Rotation

from data.toy2_so3 import MixtureIGSO3Dataset, MIGDataModule, mmd2_unbiased_R
from data.so3_interpolant import SO3Interpolant
from data import so3_utils
from data.interpolant import create_time_sampler
from data.so3_utils import *

from models.meanflow_model import ToySO3MeanFlowModel
from models import utils as mu

from experiments import utils as eu

log = eu.get_pylogger(__name__)
torch.set_float32_matmul_precision("high")


class UniformSO3Prior:
    def sample_noise_like(self, x1):
        batch_size = x1.shape[0]
        return torch.tensor(
            Rotation.random(batch_size).as_matrix(),
            device=x1.device,
            dtype=torch.float32,
        ).reshape(batch_size, 3, 3)

    def sample(self, batch_size, device):
        return torch.tensor(
            Rotation.random(batch_size).as_matrix(),
            device=device,
            dtype=torch.float32,
        ).reshape(batch_size, 3, 3)


class ToySO3SplitMeanFlowModule(LightningModule):

    def __init__(self, cfg):
        super().__init__()
        self._print_logger = logging.getLogger(__name__)
        self._exp_cfg = cfg.experiment

        self.model = ToySO3MeanFlowModel(cfg.model)

        self.prior = UniformSO3Prior()
        self.prob_path = SO3Interpolant(cfg.interpolant)

        self.model.set_interpolant(self.prob_path)

        self.fm_time_sampler = create_time_sampler(cfg.fm_time_sampler)
        self.sg_time_sampler = create_time_sampler(cfg.sg_time_sampler)

        self.save_hyperparameters()

        self.validation_samples = []

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
            "trainer/epoch_time_minutes",
            epoch_time,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self._epoch_start_time = time.time()

    def flow_matching_loss(self, batch):
        x_t = batch["xt"]
        x_1 = batch["x1"]
        t = batch["t"]

        if self._exp_cfg.training.get("use_true_vf", False):
            # Access dataset to obtain the true velocity field
            v_t = self.trainer.datamodule.dataset.true_vf(t, x_t)
        else:
            # Conditional velocity fields
            v_t = self.prob_path.cond_vf(t, x_t, x_1)

        if torch.any(torch.isnan(v_t)):
            raise ValueError("NaN encountered in v_t")

        u_t = self.model.avg_vel(x_t, t, t)

        fm_loss = torch.sum((u_t - v_t) ** 2, dim=-1)

        return fm_loss

    def semigroup_loss(self, batch):
        xt = batch["xt"]
        hat_xr = batch["hat_xr"]
        t = batch["t"]
        r = batch["r"]

        if self._exp_cfg.training.semigroup_loss_on_velocity:
            r_minus_t = torch.clamp(r - t, min=1e-4)[..., None]
            u_tgt = so3_utils.calc_rot_vf(xt, hat_xr) / r_minus_t

            u = self.model.avg_vel(xt, t, r)

            sg_loss = torch.sum((u - u_tgt) ** 2, dim=-1)
        else:
            xr = self.model.forward_flow(xt, t, r)
            sg_loss = torch.sum(so3_utils.rot_squared_dist(xr, hat_xr), dim=-1)

        return sg_loss

    def model_step(self, fm_batch, sg_batch):
        # Flow matching loss.
        fm_loss = self.flow_matching_loss(fm_batch)
        if torch.any(torch.isnan(fm_loss)):
            raise ValueError("NaN loss encountered in fm_loss")

        # Algebraic consistency.
        sg_loss = self.semigroup_loss(sg_batch)
        if torch.any(torch.isnan(sg_loss)):
            raise ValueError("NaN loss encountered in sg_loss")

        # Total loss.
        loss = (
            self._exp_cfg.training.flow_matching_loss_weight * fm_loss
            + self._exp_cfg.training.semigroup_loss_weight * sg_loss
        )
        return {
            "fm_loss": fm_loss,
            "sg_loss": sg_loss,
            "loss": loss,
        }

    def make_fm_batch(self, x1):
        t = self.fm_time_sampler(x1.shape[0], x1.device)
        x0 = self.prior.sample_noise_like(x1)
        xt = self.prob_path.sample_xt(x0, x1, t)
        return {
            "xt": xt,
            "t": t,
            "x0": x0,
            "x1": x1,
        }

    def make_sg_batch(self, x1):
        t, s, r = self.sg_time_sampler(x1.shape[0], x1.device)
        x0 = self.prior.sample_noise_like(x1)
        xt = self.prob_path.sample_xt(x0, x1, t)

        with torch.no_grad():
            hat_xs = self.model.forward_flow(xt, t, s)
            hat_xr = self.model.forward_flow(hat_xs, s, r)

        return {
            "hat_xr": hat_xr,
            "t": t,
            "r": r,
            "xt": xt,
        }

    def training_step(self, batch: Any, stage: int):
        step_start_time = time.time()

        fm_batch = self.make_fm_batch(batch["R1"])
        sg_batch = self.make_sg_batch(batch["R1"])

        try:
            batch_losses = self.model_step(fm_batch, sg_batch)
        except Exception as e:
            if isinstance(e, ValueError) and ("NaN" in str(e) or "nan" in str(e)):
                debug_dir = os.path.join(self.checkpoint_dir, "debug")
                os.makedirs(debug_dir, exist_ok=True)
                prefix = f"step_{int(self.global_step)}"

                # Try Lightning checkpoint (includes optimizer/scheduler states)
                ckpt_path = os.path.join(debug_dir, f"{prefix}_trainer.ckpt")
                self.trainer.save_checkpoint(ckpt_path)

                # Serialize input batch and model state for debugging
                def _to_cpu(obj):
                    if torch.is_tensor(obj):
                        return obj.detach().cpu()
                    if isinstance(obj, dict):
                        return {k: _to_cpu(v) for k, v in obj.items()}
                    if isinstance(obj, (list, tuple)):
                        typ = type(obj)
                        return typ(_to_cpu(x) for x in obj)
                    return obj

                batch_cpu = _to_cpu(batch)
                batch_path = os.path.join(debug_dir, f"{prefix}_noisy_batch.pt")
                model_path = os.path.join(debug_dir, f"{prefix}_model_state.pt")
                torch.save(batch_cpu, batch_path)
                torch.save(self.model.state_dict(), model_path)
                self._print_logger.error(
                    f"NaN ValueError encountered. Saved artifacts: batch={batch_path}, model={model_path}"
                )
            raise e

        total_losses = {k: torch.mean(v) for k, v in batch_losses.items()}
        for k, v in total_losses.items():
            self.log(f"train/{k}", v, on_step=True, prog_bar=False)

        # Losses to track. Stratified across t.
        for loss_name, batch_loss in batch_losses.items():
            if "fm" in loss_name:
                batch_t = fm_batch["t"]
                t_label = "t"
            elif "sg" in loss_name:
                batch_t = sg_batch["r"] - sg_batch["t"]
                t_label = "r-t"
            else:
                continue

            # Bin the loss by timestep.
            stratified_losses = mu.t_stratified_loss(
                batch_t, batch_loss, loss_name=loss_name, t_label=t_label
            )

            # Log the stratified losses.
            for k, v in stratified_losses.items():
                self.log(f"stratified_loss/{k}", v, on_step=True, prog_bar=False)

        # Training throughput
        step_time = time.time() - step_start_time
        self.log("trainer/seconds_per_batch", step_time, on_step=True, prog_bar=False)

        # This is the final training objective.
        train_loss = total_losses["loss"]
        self.log("train/loss", train_loss, on_step=True, prog_bar=False)
        return train_loss

    def validation_step(self, batch: Any, batch_idx: int):
        self.validation_samples.append(batch["R1"])

    def on_validation_epoch_end(self):
        val_samples = torch.concatenate(self.validation_samples, dim=0)

        x0 = self.prior.sample_noise_like(val_samples)

        # Multi-step samples
        multi_step_x1 = self.model.sample(x0)
        multi_step_mmd2 = mmd2_unbiased_R(val_samples, multi_step_x1)

        # One-step samples
        one_step_x1 = self.model.one_step_sample(x0)
        one_step_mmd2 = mmd2_unbiased_R(val_samples, one_step_x1)

        self.log(
            "valid/multi_step_mmd2",
            multi_step_mmd2,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.log(
            "valid/one_step_mmd2",
            one_step_mmd2,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

        self.validation_samples.clear()

        fig1 = visualize_rotation(val_samples.cpu())
        fig2 = visualize_rotation(multi_step_x1.cpu())
        fig3 = visualize_rotation(one_step_x1.cpu())
        self.logger.experiment.log(
            {
                "valid/gt_sample": wandb.Image(fig1),
                "valid/multi_step_sample": wandb.Image(fig2),
                "valid/one_step_sample": wandb.Image(fig3),
            },
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(
            params=self.model.parameters(), **self._exp_cfg.optimizer
        )

    def on_before_optimizer_step(self, optimizer):
        # Compute grad norms and rename keys for cleaner logging
        if self._exp_cfg.debug:
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

    def predict_step(self, batch, batch_idx): ...


class ToySO3MeanFlowModule(ToySO3SplitMeanFlowModule):

    def __init__(self, cfg):
        super(ToySO3SplitMeanFlowModule, self).__init__()
        self._print_logger = logging.getLogger(__name__)
        self._exp_cfg = cfg.experiment

        self.model = ToySO3MeanFlowModel(cfg.model)

        self.prior = UniformSO3Prior()
        self.prob_path = SO3Interpolant(cfg.interpolant)

        self.model.set_interpolant(self.prob_path)

        self.time_sampler = create_time_sampler(cfg.time_sampler)

        self.save_hyperparameters()

        self.validation_samples = []

        self._checkpoint_dir = None
        self._inference_dir = None

    def model_step(self, batch):
        # Flow matching loss.
        xt = batch["xt"]
        t = batch["t"]
        r = batch["r"]
        x1 = batch["x1"]

        v_rot = self.prob_path.cond_vf(t, xt, x1)
        if torch.any(torch.isnan(v_rot)):
            raise ValueError("NaN encountered in v_rot")

        tangent = (
            so3_utils.rotvec_to_tangent(xt, v_rot),
            torch.ones_like(t),
            torch.zeros_like(r),
        )
        u, du_dt = self.model.jvp_avg_vel(xt, t, r, tangent)

        assert not (du_dt.isnan()).any()

        if self._exp_cfg.training.get("GFM_MF_loss", False):
            u_tgt = ((r - t)[..., None] * du_dt + v_rot).detach()
        else:
            hat_rot_r = so3_utils.exp(xt, (r - t)[..., None] * u)
            u_tgt = (
                (r - t)[..., None] * du_dt - so3_utils.d1_log(xt, hat_rot_r, v_rot)
            ).detach()

        meanflow_loss = torch.sum((u - u_tgt) ** 2, dim=-1)

        return {
            "meanflow_loss": meanflow_loss,
        }

    def make_batch(self, x1):
        t, r = self.time_sampler(x1.shape[0], x1.device)
        x0 = self.prior.sample_noise_like(x1)
        xt = self.prob_path.sample_xt(x0, x1, t)
        return {
            "xt": xt,
            "t": t,
            "r": r,
            "x0": x0,
            "x1": x1,
        }

    def training_step(self, batch: Any, stage: int):
        step_start_time = time.time()

        batch = self.make_batch(batch["R1"])

        try:
            batch_losses = self.model_step(batch)
        except Exception as e:
            if isinstance(e, ValueError) and ("NaN" in str(e) or "nan" in str(e)):
                debug_dir = os.path.join(self.checkpoint_dir, "debug")
                os.makedirs(debug_dir, exist_ok=True)
                prefix = f"step_{int(self.global_step)}"

                # Try Lightning checkpoint (includes optimizer/scheduler states)
                ckpt_path = os.path.join(debug_dir, f"{prefix}_trainer.ckpt")
                self.trainer.save_checkpoint(ckpt_path)

                # Serialize input batch and model state for debugging
                def _to_cpu(obj):
                    if torch.is_tensor(obj):
                        return obj.detach().cpu()
                    if isinstance(obj, dict):
                        return {k: _to_cpu(v) for k, v in obj.items()}
                    if isinstance(obj, (list, tuple)):
                        typ = type(obj)
                        return typ(_to_cpu(x) for x in obj)
                    return obj

                batch_cpu = _to_cpu(batch)
                batch_path = os.path.join(debug_dir, f"{prefix}_batch.pt")
                model_path = os.path.join(debug_dir, f"{prefix}_model_state.pt")
                torch.save(batch_cpu, batch_path)
                torch.save(self.model.state_dict(), model_path)
                self._print_logger.error(
                    f"NaN ValueError encountered. Saved artifacts: batch={batch_path}, model={model_path}"
                )
            raise e

        total_losses = {k: torch.mean(v) for k, v in batch_losses.items()}
        for k, v in total_losses.items():
            self.log(f"train/{k}", v, on_step=True, prog_bar=False)

        # Losses to track. Stratified across t.
        for loss_name, batch_loss in batch_losses.items():
            batch_t = batch["r"] - batch["t"]
            t_label = "r-t"

            # Bin the loss by timestep.
            stratified_losses = mu.t_stratified_loss(
                batch_t, batch_loss, loss_name=loss_name, t_label=t_label
            )

            # Log the stratified losses.
            for k, v in stratified_losses.items():
                self.log(f"stratified_loss/{k}", v, on_step=True, prog_bar=False)

        # Training throughput
        step_time = time.time() - step_start_time
        self.log("trainer/seconds_per_batch", step_time, on_step=True, prog_bar=False)

        # This is the final training objective.
        train_loss = total_losses["meanflow_loss"]
        self.log("train/loss", train_loss, on_step=True, prog_bar=False)
        return train_loss


def visualize_rotation(rotmat: torch.Tensor, mark_start: bool = False):
    # Plot the rotation matrix as axis map on S^2.
    # Convert the rotation matrix to axis-angle representation.
    # Scatter the axes u on the sphere and color the theta.
    rotvec = rotmat_to_rotvec(rotmat)

    angle = torch.norm(rotvec, dim=-1)
    axis = rotvec / angle[..., None]

    # Plot the axes on the sphere.
    from matplotlib import pyplot as plt
    import numpy as np

    # Scatter the axes on the sphere.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(
        axis[..., 0], axis[..., 1], axis[..., 2], c=angle, cmap="viridis"
    )
    scatter.set_clim(0, np.pi)

    if mark_start:
        ax.scatter(axis[0, ..., 0], axis[0, ..., 1], axis[0, ..., 2], c="red", s=100)

    # Set colorbar to be smaller and minimize gap between plot and colorbar
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.025, pad=-0.08)
    cbar.set_label("Rotation angle")

    # Fix the aspect ratio and set a fixed view angle.
    # ax.set_aspect("equal")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Draw a very faint gray unit sphere with meridians and parallels
    contract_scale = 0.98

    # u: azimuthal, v: polar
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v)) * contract_scale
    y = np.outer(np.sin(u), np.sin(v)) * contract_scale
    z = np.outer(np.ones_like(u), np.cos(v)) * contract_scale
    # Plot the sphere
    ax.plot_surface(
        x,
        y,
        z,
        rstride=2,
        cstride=2,
        color="gray",
        edgecolor="lightgray",
        linewidth=0.2,
        alpha=0.001,  # very faint
        zorder=0,
    )
    # Draw meridians and parallels
    for phi in np.linspace(0, 2 * np.pi, 12, endpoint=False):  # Meridians
        ax.plot(
            np.cos(phi) * np.sin(v) * contract_scale,
            np.sin(phi) * np.sin(v) * contract_scale,
            np.cos(v) * contract_scale,
            color="lightgray",
            linewidth=0.5,
            alpha=0.01,
            zorder=1,
        )
    for theta in np.linspace(0, np.pi, 7)[1:-1]:  # Parallels (omit poles)
        ax.plot(
            np.cos(u) * np.sin(theta) * contract_scale,
            np.sin(u) * np.sin(theta) * contract_scale,
            np.ones_like(u) * np.cos(theta) * contract_scale,
            color="lightgray",
            linewidth=0.5,
            alpha=0.01,
            zorder=1,
        )

    ax.set_aspect("equal")
    ax.axis("off")

    return fig


from torch_ema import ExponentialMovingAverage


class EMACallback(Callback):
    def __init__(self, decay=0.999):
        self.decay = decay
        self.ema = None

    def on_train_start(self, trainer, pl_module):
        self.ema = ExponentialMovingAverage(
            pl_module.model.parameters(), decay=self.decay
        )

    def on_before_zero_grad(self, trainer, pl_module, optimizer):
        self.ema.update()

    def on_validation_epoch_start(self, trainer, pl_module):
        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.ema is not None:
            self.ema.restore()


class Experiment:

    def __init__(self, *, cfg: DictConfig):
        self.cfg = cfg
        self.exp_cfg = cfg.experiment
        self.train_device_ids = eu.get_available_device(self.exp_cfg.num_devices)
        log.info(f"Training with devices: {self.train_device_ids}")

        self.setup_dataset()

        self.datamodule: LightningDataModule = MIGDataModule(
            train_dataset=self.train_dataset,
            val_dataset=self.valid_dataset,
            batch_size=self.exp_cfg.batch_size,
        )

        if self.exp_cfg.loss_type == "meanflow":
            self.module: LightningModule = ToySO3MeanFlowModule(self.cfg)
        elif self.exp_cfg.loss_type == "semigroup":
            self.module: LightningModule = ToySO3SplitMeanFlowModule(self.cfg)
        else:
            raise ValueError(f"Unrecognized loss type: {self.exp_cfg.loss_type}")

    def setup_dataset(self):
        self.train_dataset = MixtureIGSO3Dataset(
            N=10000, device=f"cuda:{self.train_device_ids[0]}", seed=self.exp_cfg.seed
        )
        self.valid_dataset = MixtureIGSO3Dataset(
            N=512,
            device=f"cuda:{self.train_device_ids[0]}",
            seed=self.exp_cfg.seed + 10,
        )

    def train(self):
        callbacks = []
        if self.exp_cfg.debug:
            log.info("Debug mode.")
            logger = None
        else:
            logger = WandbLogger(
                **self.exp_cfg.wandb,
            )

            # Checkpoint directory.
            ckpt_dir = self.exp_cfg.checkpointer.dirpath
            os.makedirs(ckpt_dir, exist_ok=True)
            log.info(f"Checkpoints saved to {ckpt_dir}")

            # Model checkpoints
            callbacks.append(ModelCheckpoint(**self.exp_cfg.checkpointer))

            # EMA
            if self.exp_cfg.use_ema:
                callbacks.append(EMACallback(decay=self.exp_cfg.ema_decay))

            # Save config only for main process.
            cfg_path = os.path.join(ckpt_dir, "config.yaml")
            with open(cfg_path, "w") as f:
                OmegaConf.save(config=self.cfg, f=f.name)
            cfg_dict = OmegaConf.to_container(self.cfg, resolve=True)
            flat_cfg = dict(eu.flatten_dict(cfg_dict))
            logger.experiment.config.update(flat_cfg)

        trainer = Trainer(
            **self.exp_cfg.trainer,
            callbacks=callbacks,
            logger=logger,
            use_distributed_sampler=False,
            enable_progress_bar=True,
            enable_model_summary=True,
            devices=self.train_device_ids,
        )
        trainer.fit(
            model=self.module,
            datamodule=self.datamodule,
        )


@hydra.main(
    version_base=None, config_path="../configs", config_name="base_toy_so3.yaml"
)
def main(cfg: DictConfig):
    torch.autograd.set_detect_anomaly(mode=True)
    seed_everything(cfg.experiment.seed)
    exp = Experiment(cfg=cfg)
    exp.train()


if __name__ == "__main__":
    main()
