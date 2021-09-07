#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from DataSelection.datasets.cifar10 import get_cifar10_label_names, load_cifar10_file, download_cifar10_data, plot_images
from DataSelection.utils.plot import plot_confusion_matrix, plot_label_entropy_histogram, \
    plot_label_entropy_cumsum
from DataSelection.datasets.cifar10h import CIFAR10H
from DataSelection.evaluation.metrics import compute_label_entropy


def main() -> None:
    label_names = get_cifar10_label_names()
    path_cifar10_test_batch = download_cifar10_data()
    d_test_complete = CIFAR10H.download_cifar10h_labels()
    test_batch = load_cifar10_file(path_cifar10_test_batch)

    # Confusion matrix
    y_true = list()
    y_pred = list()
    for sample_row_id in range(d_test_complete.shape[0]):
        sample_row = d_test_complete[sample_row_id]
        for _iter, _el in enumerate(sample_row.tolist()):
            y_pred.extend([_iter] * _el)
        y_true.extend([np.argmax(sample_row)] * np.sum(sample_row))
    plot_confusion_matrix(y_true, y_pred, label_names)

    # Histogram of target entropies (measure of difficulty or ambiguity)
    plot_cumulative = True
    label_entropy = compute_label_entropy(d_test_complete)
    if plot_cumulative:
        plot_label_entropy_cumsum(label_entropy, num_bins=20)
    else:
        plot_label_entropy_histogram(label_entropy, num_bins=20)
    plt.show()

    # Display samples with entropy larger than a threshold value
    threshold = 0.35
    ambigious_case_ids = np.where(label_entropy > threshold)[0].tolist()
    screenshot_save_path = Path(__file__).parent.absolute() / "experiments/screenshots/"
    screenshot_save_path.mkdir(parents=True, exist_ok=True)
    plot_images(images=test_batch[b'data'],
                selected_sample_ids=ambigious_case_ids,
                cifar10h_labels=d_test_complete,
                label_names=label_names,
                save_directory=screenshot_save_path)


if __name__ == "__main__":
    main()
