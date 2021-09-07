from typing import Any

import numpy as np
import torch
import yacs.config
from torch.utils.data import DataLoader
from torchvision.datasets.vision import VisionDataset


def get_number_of_samples_per_epoch(dataloader: torch.utils.data.DataLoader) -> int:
    """
    Returns the expected number of samples for a single epoch
    """
    total_num_samples = len(dataloader.dataset)
    batch_size = dataloader.batch_size
    drop_last = dataloader.drop_last
    num_samples = int(total_num_samples / batch_size) * batch_size if drop_last else total_num_samples

    return num_samples


def get_train_dataloader(train_dataset: VisionDataset,
                         config: yacs.config.CfgNode,
                         seed: int,
                         **kwargs: Any) -> DataLoader:
    return torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        num_workers=config.train.dataloader.num_workers,
        pin_memory=config.train.dataloader.pin_memory,
        worker_init_fn=WorkerInitFunc(seed),
        **kwargs)


class WorkerInitFunc:
    def __init__(self, seed: int) -> None:
        self.seed = seed

    def __call__(self, worker_id: int) -> None:
        return np.random.seed(self.seed + worker_id)


def get_val_dataloader(val_dataset: VisionDataset, config: yacs.config.CfgNode, seed: int) -> DataLoader:
    return torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.validation.batch_size,
        shuffle=False,
        num_workers=config.validation.dataloader.num_workers,
        pin_memory=config.validation.dataloader.pin_memory,
        worker_init_fn=WorkerInitFunc(seed))
