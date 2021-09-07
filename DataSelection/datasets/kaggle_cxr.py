#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import PIL
import numpy as np
import pandas as pd
import pydicom as dicom
from PIL import Image
from torch.utils.data import Dataset

KAGGLE_TOTAL_SIZE = 26684


class KaggleCXR(Dataset):
    def __init__(self,
                 data_directory: str,
                 use_training_split: bool,
                 train_fraction: float = 0.8,
                 seed: int = 1234,
                 shuffle: bool = True,
                 transform: Optional[Callable] = None,
                 num_samples: int = None,
                 return_index: bool = True) -> None:
        """
        Class for the full Kaggle RSNA Pneumonia Detection Dataset.

        :param data_directory: the directory containing all training images from the Challenge (stage 1) as well as the
        dataset.csv containing the kaggle and the original labels.
        :param use_training_split: whether to return the training or the validation split of the dataset.
        :param train_fraction: the proportion of samples to use for training
        :param seed: random seed to use for dataset creation
        :param shuffle: whether to shuffle the dataset prior to spliting between validation and training
        :param transform: a preprocessing function that takes a PIL image as input and returns a tensor
        :param num_samples: number of the samples to return (has to been smaller than the dataset split)
        """
        self.data_directory = Path(data_directory)
        if not self.data_directory.exists():
            logging.error(
                f"The data directory {self.data_directory} does not exist. Make sure to download to Kaggle data "
                f"first.The kaggle dataset can "
                "be acceded via the Kaggle CLI kaggle competitions download -c rsna-pneumonia-detection-challenge or "
                "on the main page of the challenge "
                "https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/data?select=stage_2_train_images")

        self.train = use_training_split
        self.train_fraction = train_fraction
        self.seed = seed
        self.random_state = np.random.RandomState(seed)
        full_dataset = pd.read_csv(self.data_directory / "dataset.csv")
        self.dataset_dataframe = full_dataset
        self.transforms = transform

        targets = self.dataset_dataframe.label.values.astype(np.int64)
        subjects_ids = self.dataset_dataframe.subject.values

        self.num_classes = 2
        self.num_datapoints = len(self.dataset_dataframe)
        all_indices = np.arange(len(self.dataset_dataframe))

        # ------------- Split the data into training and validation sets ------------- #
        num_samples_set1 = int(self.num_datapoints * self.train_fraction)
        sampled_indices = self.random_state.permutation(all_indices) \
            if shuffle else all_indices
        train_indices = sampled_indices[:num_samples_set1]
        val_indices = sampled_indices[num_samples_set1:]
        self.indices = train_indices if use_training_split else val_indices

        # ------------- Select subset of current split ------------- #
        if num_samples is not None:
            assert 0 < num_samples <= len(self.indices)
            self.indices = self.indices[:num_samples]

        self.subject_ids = subjects_ids[self.indices]

        self.targets = targets[self.indices].reshape(-1)

        dataset_type = "TRAIN" if use_training_split else "VAL"
        logging.info(f"Proportion of positive labels - {dataset_type}: {np.mean(self.targets)}")
        logging.info(f"Number samples - {dataset_type}: {self.targets.shape[0]}")
        self.return_index = return_index

    def __getitem__(self, index: int) -> Union[Tuple[int, PIL.Image.Image, int], Tuple[PIL.Image.Image, int]]:
        """

        :param index: The index of the sample to be fetched
        :return: The image and label tensors
        """
        subject_id = self.subject_ids[index]
        filename = self.data_directory / f"{subject_id}.dcm"
        target = self.targets[index]
        scan_image = dicom.dcmread(filename).pixel_array
        scan_image = Image.fromarray(scan_image)
        if self.transforms is not None:
            scan_image = self.transforms(scan_image)
        if self.return_index:
            return index, scan_image, int(target)
        return scan_image, int(target)

    def __len__(self) -> int:
        """

        :return: The size of the dataset
        """
        return len(self.indices)

    def get_label_names(self) -> List[str]:
        return ["Normal", "Opacity"]


if __name__ == "__main__":
    from DataSelection.deep_learning.create_dataset_transforms import create_chest_xray_transform
    from DataSelection.deep_learning.utils import load_ssl_model_config
    from DataSelection.utils.default_paths import PROJECT_ROOT_DIR
    import matplotlib.pyplot as plt

    config_path = PROJECT_ROOT_DIR / r"DataSelection/deep_learning/self_supervised/configs" \
                                     r"/kaggle_simclr.yaml"
    config = load_ssl_model_config(Path(config_path))
    transform = create_chest_xray_transform(config, is_train=True)
    dataset = KaggleCXR(data_directory=config.dataset.dataset_dir, use_training_split=True,
                        transform=transform,
                        return_index=False)
    fig, ax = plt.subplots(5, 5, figsize=(15, 15))
    ax = ax.ravel()
    k = 50
    for i in range(25):
        scan_image, label = dataset.__getitem__(i + k)  # type: ignore
        ax[i].imshow(np.transpose(scan_image, [1, 2, 0]))
        ax[i].set_title(dataset.get_label_names()[label])
        ax[i].set_axis_off()
    plt.show()
