#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import argparse
from typing import Any, Callable, List

import matplotlib.pyplot as plt
import numpy as np
import torchvision
from torchvision.transforms import ToTensor

from DataSelection.deep_learning.transforms import ExpandChannels
from DataSelection.deep_learning.utils import load_selector_config
from DataSelection.selection.data_curation_utils import get_user_specified_selectors
from DataSelection.utils.dataset_utils import load_dataset_and_initial_labels_for_simulation
from DataSelection.utils.default_paths import DATA_CURATION_DIR
from DataSelection.utils.generic import create_folder, get_data_curation_parser, get_logger
from DataSelection.utils.plot import plot_latent_space_and_noise_cases

"""
This script takes a config as argument, loads a trained model from checkpoint and runs the data
selection pipeline on the training set. The selected samples are returned by batchs of 50 images
to review. The images are saved as thumbnails (one file per batch) along with a csv identifing these images
in the original dataset (one per batch). The results can be found in the DATA_CURATION_DIR. The number of
samples to relabel can be specified in the selector config via the "number_samples_to_relabel2 parameter.
"""

def get_plotting_transform() -> Callable:
    transforms: List[Any] = [torchvision.transforms.Resize(256)]
    transforms += [ToTensor(), ExpandChannels()]
    return torchvision.transforms.Compose(transforms)

def main(args: argparse.Namespace) -> None:
    dataset, initial_labels = load_dataset_and_initial_labels_for_simulation(args.config[0], on_val_set=False)
    label_names = dataset.get_label_names()
    config = load_selector_config(args.config[0])
    number_relabels = config.selector.number_samples_to_relabel
    user_specified_selectors = get_user_specified_selectors(list_configs=args.config,
                                                            dataset=dataset,
                                                            output_path=DATA_CURATION_DIR,
                                                            plot_embeddings=False)
    sample_selectors = {**user_specified_selectors}
    batch_size = 49
    num_batches = number_relabels // batch_size
    for name, sample_selector in sample_selectors.items():
        create_folder(DATA_CURATION_DIR / name)
        for i in range(num_batches):
            sample_ids = sample_selector.get_batch_of_samples_to_annotate(current_labels=initial_labels,
                                                                          max_cases=batch_size)
            selected_df = dataset.get_selected_df(sample_ids)
            selected_df.to_csv(DATA_CURATION_DIR / name / f"selected_{i}.csv", index=False)
            plot_selected_images(dataset, sample_ids, label_names, name, i)
            if i == 0 and sample_selector.embeddings is not None:
                plot_embedding_selected(initial_labels[:, 1], sample_ids, sample_selector.embeddings, name)


def plot_embedding_selected(initial_labels: np.ndarray, sample_ids: np.array, embeddings: np.ndarray,
                            name: str) -> None:
    """
    Plot selected samples in embeddings space
    :param initial_labels:
    :param sample_ids:
    :param embeddings:
    :param name:
    :return:
    """
    selected = np.zeros_like(initial_labels)
    selected[sample_ids] = 1
    fig, ax = plt.subplots(1, 1)
    plot_latent_space_and_noise_cases(ax=ax,
                                      embeddings=embeddings,
                                      labels=initial_labels,
                                      indicator_noisy_labels=np.ones(initial_labels.shape[0]),
                                      selected_cases=selected)
    plt.savefig(DATA_CURATION_DIR / f"embeddings_{name}.png")


def plot_selected_images(dataset: Any, sample_ids: np.ndarray, label_names: List, name: str, batch: int = 0) -> None:
    """
    Show grid of images of selected samples
    :param dataset:
    :param sample_ids:
    :param label_names:
    :param name:
    :return:
    """
    nrows = sample_ids.size // 7 if sample_ids.size % 7 == 0 else sample_ids.size // 7 + 1
    fig, ax = plt.subplots(nrows, 7, figsize=(15, 20))
    ax = ax.ravel()
    t = dataset.transforms
    dataset.transforms = get_plotting_transform()
    for i in range(sample_ids.shape[0]):
        id, scan_image, label = dataset.__getitem__(sample_ids[i])
        ax[i].imshow(np.transpose(scan_image, [1, 2, 0]))
        ax[i].set_title(f"{label_names[label]}\n{id}")
        ax[i].set_axis_off()
    dataset.transforms = t
    plt.show()
    plt.savefig(DATA_CURATION_DIR / name / f"thumbnails_{batch}.png")
    plt.close()


if __name__ == '__main__':
    create_folder(DATA_CURATION_DIR)
    get_logger(DATA_CURATION_DIR / 'dataread.log')
    parser = get_data_curation_parser()
    args, unknown_args = parser.parse_known_args()
    main(args)
