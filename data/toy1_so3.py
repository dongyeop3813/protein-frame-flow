"""Copyright (c) Dreamfold."""

from torch import Tensor
import torch
from torch.utils.data import Dataset
import numpy as np
import lightning as L
from torch.utils.data import DataLoader
from data.so3_utils import geodesic_dist


def _get_split(data, split, seed):
    assert split in ["train", "valid", "test", "all"], f"split {split} not supported."
    if split != "all":
        rng = np.random.default_rng(seed)
        indices = np.arange(len(data))
        rng.shuffle(indices)

        n = len(data)
        if split == "train":
            data = data[indices[: int(n * 0.8)]]
        elif split == "valid":
            data = data[indices[int(n * 0.8) : int(n * 0.9)]]
        elif split == "test":
            data = data[indices[int(n * 0.9) :]]
    return data


class SpecialOrthogonalGroup(Dataset):
    def __init__(self, root="./data", split="train", seed=12345):
        data = np.load(f"{root}/orthogonal_group.npy").astype("float32")
        self.data = _get_split(data, split, seed)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class SO3DataModule(L.LightningDataModule):
    def __init__(self, root="data", batch_size=512, num_workers=1, seed=12345):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

    def setup(self, stage=None):
        self.train_dataset = SpecialOrthogonalGroup(
            root=self.root, split="train", seed=self.seed
        )
        self.val_dataset = SpecialOrthogonalGroup(
            root=self.root, split="valid", seed=self.seed
        )
        self.test_dataset = SpecialOrthogonalGroup(
            root=self.root, split="test", seed=self.seed
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def get_test_tensor(self) -> Tensor:
        return torch.from_numpy(self.test_dataset.data)

    def get_val_tensor(self) -> Tensor:
        return torch.from_numpy(self.val_dataset.data)


@torch.inference_mode()
def compute_mmd(x: Tensor, y: Tensor, gamma: float = 1.0) -> float:
    assert x.shape[0] == y.shape[0]

    def kernel(a: Tensor, b: Tensor) -> Tensor:
        ds = geodesic_dist(a, b).square_()
        ds = (-gamma * ds).exp()
        return ds

    # calculate inner x, inner y, cross x and y
    x0 = x.repeat((x.shape[0],) + (1,) * (x.ndim - 1))
    x1 = x.repeat_interleave(x.shape[0], dim=0)
    y0 = y.repeat((y.shape[0],) + (1,) * (y.ndim - 1))
    y1 = y.repeat_interleave(y.shape[0], dim=0)
    inner_x = kernel(x0, x1)
    inner_y = kernel(y0, y1)
    cross_xy = kernel(x0, y1)
    return (inner_x + inner_y - 2 * cross_xy).mean().sqrt().item()


if __name__ == "__main__":
    dm = SO3DataModule()
    dm.setup()
    for batch in dm.train_dataloader():
        print(batch.shape)
        import ipdb

        ipdb.set_trace()
