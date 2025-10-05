import random
from typing import Optional, Sequence

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sgfm.common.utils import PROJECT_ROOT 
from sgfm.pl_data.dataset import cryst_collate_fn


def worker_init_fn(id: int):
    """
    DataLoaders workers init function.

    Initialize the numpy.random seed correctly for each worker, so that
    random augmentations between workers and/or epochs are not identical.

    If a global seed is set, the augmentations are deterministic.

    https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    """
    uint64_seed = torch.initial_seed()
    ss = np.random.SeedSequence([uint64_seed])
    # More than 128 bits (4 32-bit words) would be overkill.
    np.random.seed(ss.generate_state(4))
    random.seed(uint64_seed)


class CrystDataModule(pl.LightningDataModule):
    def __init__(
        self,
        datasets: DictConfig,
        num_workers: DictConfig,
        batch_size: DictConfig,
    ):
        super().__init__()
        self.datasets = datasets
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.collate_fn = cryst_collate_fn

        self.train_dataset: Optional[Dataset] = None
        self.val_datasets: Optional[Sequence[Dataset]] = None
        self.test_datasets: Optional[Sequence[Dataset]] = None


    def prepare_data(self) -> None:
        # download only
        pass

    def setup(self, stage: Optional[str] = None):
        """
        construct datasets
        """
        if stage is None or stage == "fit":
            self.train_dataset = hydra.utils.instantiate(self.datasets.train)
            print('start val datasets')
            self.val_datasets = [
                hydra.utils.instantiate(dataset_cfg)
                for dataset_cfg in self.datasets.val
            ]

        if stage is None or stage == "test":
            self.test_datasets = [
                hydra.utils.instantiate(dataset_cfg)
                for dataset_cfg in self.datasets.test
            ]

    def train_dataloader(self, shuffle = True, subset_inds: Optional[list[int]] = None) -> DataLoader:
        if subset_inds is None:
            dataset = self.train_dataset
        else:
            dataset = torch.utils.data.Subset(self.train_dataset, subset_inds)
        return DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=self.batch_size.train,
            num_workers=self.num_workers.train,
            worker_init_fn=worker_init_fn,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self, subset_inds: Optional[list[list[int]]] = None) -> Sequence[DataLoader]:
        if subset_inds is None:
            datasets = self.val_datasets
        else:
            datasets = [torch.utils.data.Subset(dataset, subinds) for dataset, subinds in zip(self.val_datasets, subset_inds)]
        return [
            DataLoader(
                dataset,
                shuffle=False,
                batch_size=self.batch_size.val,
                num_workers=self.num_workers.val,
                worker_init_fn=worker_init_fn,
                collate_fn=self.collate_fn
            )
            for dataset in datasets
        ]

    def test_dataloader(self, subset_inds: Optional[list[list[int]]] = None) -> Sequence[DataLoader]:
        if subset_inds is None:
            datasets = self.test_datasets
        else:
            datasets = [torch.utils.data.Subset(dataset, subinds) for dataset, subinds in zip(self.test_datasets, subset_inds)]
        return [
            DataLoader(
                dataset,
                shuffle=False,
                batch_size=self.batch_size.test,
                num_workers=self.num_workers.test,
                worker_init_fn=worker_init_fn,
                collate_fn=self.collate_fn
            )
            for dataset in datasets
        ]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.datasets=}, "
            f"{self.num_workers=}, "
            f"{self.batch_size=})"
        )
