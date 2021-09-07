#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import PIL
import numpy as np
import pandas as pd
import pydicom as dicom
from PIL import Image
from torch.utils.data import Dataset

from DataSelection.datasets.label_distribution import LabelDistribution
from DataSelection.selection.simulation_statistics import SimulationStats
from DataSelection.utils.generic import convert_labels_to_one_hot


class NoisyKaggleSubsetCXR(Dataset):
    def __init__(self, data_directory: str,
                 use_training_split: bool,
                 train_fraction: float = 0.9,
                 seed: int = 1234,
                 shuffle: bool = True,
                 transform: Optional[Callable] = None,
                 num_samples: Optional[int] = None,
                 use_noisy_fixed_labels: bool = True) -> None:
        """
        Class for the noisy Kaggle RSNA Pneumonia Detection Dataset. This dataset uses the kaggle dataset with noisy labels
        as the original labels from RSNA and the clean labels are the Kaggle labels. This dataset is a subsample from
        the full Kaggle set containing ~11k samples.

        :param data_directory: the directory containing all training images from the Challenge (stage 1) as well as the
        dataset.csv containing the kaggle and the original labels.
        :param use_training_split: whether to return the training or the validation split of the dataset.
        :param train_fraction: the proportion of samples to use for training
        :param seed: random seed to use for dataset creation
        :param shuffle: whether to shuffle the dataset prior to spliting between validation and training
        :param transform: a preprocessing function that takes a PIL image as input and returns a tensor
        :param num_samples: number of the samples to return (has to been smaller than the dataset split)
        :param use_noisy_fixed_labels: if True use the original labels as the initial labels else use the clean labels.
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
        self.dataset_dataframe = pd.read_csv(str(Path(__file__).parent / "noisy_kaggle_dataset.csv"))
        self.transforms = transform

        # ------------- Clean original dataset ------------- #
        # Restrict our dataset to the following classes:
        # Kaggle positive & NIH No Finding (noise)
        # Kaggle positive & NIH Pneumonia
        # Kaggle positive & NIH Consolidation or Infiltration (Pneumonia like opacities)
        # Kaggle negative & NIH No Finding
        # Kaggle negative & NIH pneumonia (noise)
        self.dataset_dataframe = self.dataset_dataframe[
            (self.dataset_dataframe.label_orig_pneumonia_like) | (self.dataset_dataframe.no_finding)]
        self.dataset_dataframe = self.dataset_dataframe[
            (self.dataset_dataframe.label_kaggle) | (
                    self.dataset_dataframe.label_orig_pneumonia | self.dataset_dataframe.no_finding)]

        orig_labels = self.dataset_dataframe.label_orig_pneumonia_like.values.astype(np.int64).reshape(-1, 1)
        kaggle_labels = self.dataset_dataframe.label_kaggle.values.astype(np.int64).reshape(-1, 1)
        subjects_ids = self.dataset_dataframe.subject.values
        self.kaggle_labels = kaggle_labels

        # Convert clean labels to one-hot to populate label counts
        one_hot_kaggle_labels = convert_labels_to_one_hot(kaggle_labels, n_classes=2)
        _, self.num_classes = one_hot_kaggle_labels.shape
        assert self.num_classes == 2

        self.num_datapoints = len(self.dataset_dataframe)
        all_indices = np.arange(self.num_datapoints)

        # ------------- Split the data into training and validation sets ------------- #
        num_samples_set1 = int(self.num_datapoints * self.train_fraction)
        all_indices = self.random_state.permutation(all_indices) \
            if shuffle else all_indices
        train_indices = all_indices[:num_samples_set1]
        val_indices = all_indices[num_samples_set1:]
        self.indices = train_indices if use_training_split else val_indices

        # ------------- Select subset of current split ------------- #
        # If n_samples is set to a value < TOTAL_NOISY_KAGGLE_DATASET_SIZE i.e. for data_curation
        num_samples = self.num_datapoints if num_samples is None else num_samples
        if num_samples < self.num_datapoints:
            assert 0 < num_samples <= len(self.indices)
        self.indices = self.indices[:num_samples]
        self.subject_ids = subjects_ids[self.indices]

        # Label distribution is constructed from the true labels
        self.label_counts = one_hot_kaggle_labels[self.indices]
        self.label_distribution = LabelDistribution(seed, self.label_counts)

        self.orig_labels = orig_labels[self.indices].reshape(-1)
        self.kaggle_labels = self.kaggle_labels[self.indices].reshape(-1)

        self.targets = self.orig_labels if use_noisy_fixed_labels else self.kaggle_labels
        self.targets = self.targets.reshape(-1)

        # Identify case ids for ambiguous and clear label noise cases
        label_stats = SimulationStats(name="NoisyKaggle", true_label_counts=self.label_counts,
                                      initial_labels=convert_labels_to_one_hot(self.targets, self.num_classes))
        self.clear_mislabeled_cases = label_stats.mislabelled_not_ambiguous_sample_ids[0]
        self.ambiguity_metric_args: Dict = {"clear_mislabelled_ids": self.clear_mislabeled_cases}
        self.num_samples = self.targets.shape[0]
        dataset_type = "TRAIN" if use_training_split else "VAL"
        logging.info(f"Proportion of positive clean labels - {dataset_type}: {np.mean(self.kaggle_labels)}")
        logging.info(f"Proportion of positive noisy labels - {dataset_type}: {np.mean(self.targets)}")
        logging.info(f"Noise rate on the {dataset_type} dataset: {np.mean(self.kaggle_labels != self.targets)} \n")

    def __getitem__(self, index: int) -> Tuple[int, PIL.Image.Image, int]:
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
        if scan_image.shape == 2:
            scan_image = scan_image.unsqueeze(dim=0)
        return index, scan_image, int(target)

    def __len__(self) -> int:
        """

        :return: The size of the dataset
        """
        return len(self.indices)

    def get_label_names(self) -> List[str]:
        return ["Normal", "Opacity"]
