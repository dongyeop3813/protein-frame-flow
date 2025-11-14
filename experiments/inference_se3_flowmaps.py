"""
Script for running inference and evaluation.

Load checkpoint and run sampling for later evaluation (designability).
"""

import os
import time
import numpy as np
import hydra
import torch
import GPUtil
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from experiments import utils as eu
from analysis import utils as au

from models.splitmf_module import SplitMeanFlowModule
from models.meanflow_module import MeanFlowModule


torch.set_float32_matmul_precision("high")
torch.set_grad_enabled(False)
log = eu.get_pylogger(__name__)


class EvalRunner:

    def __init__(self, cfg: DictConfig):
        """Initialize sampler.

        Args:
            cfg: inference config.
        """
        ckpt_path = cfg.ckpt_path
        ckpt_dir = os.path.dirname(ckpt_path)
        ckpt_cfg = OmegaConf.load(os.path.join(ckpt_dir, "config.yaml"))

        self.cfg = cfg
        self.ckpt_cfg = ckpt_cfg
        self.sample_cfg = cfg.samples
        self.rng = np.random.default_rng(cfg.seed)

        # Set-up output directory
        if cfg.name is not None:
            self.output_dir = os.path.join(cfg.output_dir, cfg.name)
        else:
            datetime_str = f"run_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
            self.output_dir = os.path.join(cfg.output_dir, datetime_str)
        os.makedirs(self.output_dir, exist_ok=True)

        # Read checkpoint and initialize module.
        if cfg.module_type == "splitmf":
            self.module = SplitMeanFlowModule.load_from_checkpoint(
                checkpoint_path=ckpt_path, cfg=ckpt_cfg
            )
        elif cfg.module_type == "meanflow":
            self.module = MeanFlowModule.load_from_checkpoint(
                checkpoint_path=ckpt_path, cfg=ckpt_cfg
            )
        else:
            raise ValueError(f"Unsupported module type: {cfg.module_type}")

        log.info(pl.utilities.model_summary.ModelSummary(self.module))
        self.module.eval()
        self.model = self.module.model

    def get_device(self):
        devices = GPUtil.getAvailable(order="memory", limit=8, excludeID=[0, 1, 2])
        return devices[0]

    def run_sampling(self):
        device = self.get_device()
        self.module.to(device)
        self.module.interpolant.set_device(device)

        log.info(f"Using device: {device}")
        log.info(f"Sampling from flow maps")

        all_sample_lengths = range(
            self.sample_cfg.min_length,
            self.sample_cfg.max_length + 1,
            self.sample_cfg.length_step,
        )
        num_batch = self.sample_cfg.samples_per_length

        for length in all_sample_lengths:
            samples = self.sample(num_batch, length)
            length_dir = os.path.join(self.output_dir, f"length_{length}")
            os.makedirs(length_dir, exist_ok=True)
            for sample_id in range(num_batch):
                sample_dir = os.path.join(length_dir, f"sample_{sample_id}")
                os.makedirs(sample_dir, exist_ok=True)
                pdb_path = os.path.join(sample_dir, "sample.pdb")

                final_pos = samples[sample_id]
                au.write_prot_to_pdb(final_pos, pdb_path, no_indexing=True)

            log.info(f"Done writing samples for length {length}")
        log.info(f"Finished sampling")

    def sample(self, num_batch, num_res):
        if self.cfg.one_step_evaluation:
            samples = self.module.one_step_sample(num_batch, num_res)
        else:
            samples = self.module.multi_step_sample(num_batch, num_res)

        return samples


@hydra.main(
    version_base=None, config_path="../configs", config_name="flowmap_inference"
)
def run(cfg: DictConfig) -> None:

    # Read model checkpoint.
    log.info(f"Starting inference")
    start_time = time.time()
    sampler = EvalRunner(cfg)
    sampler.run_sampling()
    elapsed_time = time.time() - start_time
    log.info(f"Finished in {elapsed_time:.2f}s")


if __name__ == "__main__":
    run()
