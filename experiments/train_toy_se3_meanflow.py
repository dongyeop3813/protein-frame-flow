import os
import torch
import hydra
from omegaconf import DictConfig, OmegaConf

# Pytorch lightning imports
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from data.toy_data import ToySE3, ToySE3DataModule
from models.meanflow_module import MeanFlowModule
from experiments import utils as eu

log = eu.get_pylogger(__name__)
torch.set_float32_matmul_precision("high")


class Experiment:

    def __init__(self, *, cfg: DictConfig):
        self._cfg = cfg
        self._data_cfg = cfg.data
        self._exp_cfg = cfg.experiment
        self._task = self._data_cfg.task
        self._setup_dataset()
        self._datamodule: LightningDataModule = ToySE3DataModule(
            data_cfg=self._data_cfg,
            train_dataset=self._train_dataset,
            valid_dataset=self._valid_dataset,
        )
        self._module: LightningModule = MeanFlowModule(self._cfg)

    def _setup_dataset(self):
        self._train_dataset = ToySE3()
        self._valid_dataset = ToySE3()

    def train(self):
        callbacks = []
        if self._exp_cfg.debug:
            log.info("Debug mode.")
            logger = None
        else:
            logger = WandbLogger(
                **self._exp_cfg.wandb,
            )

            # Checkpoint directory.
            ckpt_dir = self._exp_cfg.checkpointer.dirpath
            os.makedirs(ckpt_dir, exist_ok=True)
            log.info(f"Checkpoints saved to {ckpt_dir}")

            # Model checkpoints
            callbacks.append(ModelCheckpoint(**self._exp_cfg.checkpointer))

            # Save config only for main process.
            cfg_path = os.path.join(ckpt_dir, "config.yaml")
            with open(cfg_path, "w") as f:
                OmegaConf.save(config=self._cfg, f=f.name)

        trainer = Trainer(
            **self._exp_cfg.trainer,
            callbacks=callbacks,
            logger=logger,
            enable_progress_bar=True,
            enable_model_summary=True,
        )
        trainer.fit(
            model=self._module,
            datamodule=self._datamodule,
        )


@hydra.main(
    version_base=None, config_path="../configs", config_name="base_toy_meanflow.yaml"
)
def main(cfg: DictConfig):
    exp = Experiment(cfg=cfg)
    exp.train()


if __name__ == "__main__":
    main()
