from typing import Any
import torch
import time
import os
import logging
import torch.distributed as dist
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import grad_norm


class Timer:
    def __init__(self):
        self.start_time = time.time()

    def second(self):
        return time.time() - self.start_time

    def minute(self):
        return (time.time() - self.start_time) / 60.0


class TrainingEpochTimerHook(Callback):

    def on_train_epoch_start(self, trainer, pl_module):
        self.timer = Timer()

    def on_train_epoch_end(self, trainer, pl_module):
        epoch_time = self.timer.minute()
        pl_module.log(
            "train/epoch_time_minutes",
            epoch_time,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )


class LightningModuleWrapper(LightningModule):
    """
    Lightning module wrapper with utilities functions.

        1. Set the checkpoint and inference directories for distributed training.
        2. Measure the minute per epoch training time.
        3. Provide utilities functions for the underlying module.
    """

    def __init__(self):
        super().__init__()
        self.print_logger = logging.getLogger(__name__)
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

    def configure_callbacks(self):
        return [TrainingEpochTimerHook()]

    def log_scalar(
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
        """
        Log function for scalar values.
        It is a wrapper around the LightningModule.log function.
        It prevents the user from accidentally syncing dist when rank_zero_only=True.
        """
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

    def make_timer(self):
        return Timer()

    def log_on_exception(self, input: Any):
        """
        Log on exception. Save the input batch and model state for debugging.
        """
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

        input_cpu = _to_cpu(input)
        batch_path = os.path.join(debug_dir, f"{prefix}_input_batch.pt")
        model_path = os.path.join(debug_dir, f"{prefix}_model_state.pt")
        torch.save(input_cpu, batch_path)
        torch.save(self.model.state_dict(), model_path)
        self.print_logger.error(
            f"NaN ValueError encountered. Saved artifacts: input_batch={batch_path}, model={model_path}"
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

    @property
    def training_cfg(self):
        return self._exp_cfg.training
