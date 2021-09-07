#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from collections import Callable
from pathlib import Path
from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torchvision
from PIL import Image
from torchvision.transforms import ToTensor

from DataSelection.datasets.covid_cxr import CovidCXR
from DataSelection.deep_learning.transforms import ExpandChannels, ToNumpy
from DataSelection.utils.default_paths import DATA_CURATION_DIR
from DataSelection.utils.generic import create_folder


def get_transform() -> Callable:
    transforms: List[Any] = [torchvision.transforms.Resize(256)]
    transforms += [ToNumpy()]
    transforms.append(ExpandChannels())
    transforms.append(ToTensor())
    return torchvision.transforms.Compose(transforms)


def get_image(scan_id: str, df: pd.DataFrame, transform: Callable, data_directory: Path) -> Tuple[np.ndarray, int]:
    """
    :param scan_id: The index of the sample to be fetched
    :param df: dataframe with the dataset
    :param transform: transformation
    :param: data directory
    :return: The image and label tensors
    """
    subj = df.loc[df.series == scan_id]
    folder_id = subj.subject.values[0]
    target = subj.label.values[0]
    filename = data_directory / folder_id / scan_id / "CR.dcm"
    scan_image = CovidCXR.load_dicom_image(filename)[0]
    max, min = scan_image.max(), scan_image.min()
    scan_image = (scan_image - min) / max
    scan_image = Image.fromarray(scan_image)
    if transform is not None:
        scan_image = transform(scan_image)
    if scan_image.shape == 2:
        scan_image = scan_image.unsqueeze(dim=0)
    return scan_image, int(target)


if __name__ == "__main__":
    create_folder(DATA_CURATION_DIR / "excluded_no_processing")
    df = pd.read_csv(r"/datadrive/uhb/excluded_cases_v2.csv")
    scan_ids = df.series.values
    transform = get_transform()
    label_names = ["Normal", "Covid"]
    data_directory = Path(r"/datadrive/uhb")
    batch_size = 49
    num_batches = scan_ids.size // batch_size
    for batch in range(num_batches):
        sample_ids = scan_ids[batch * batch_size: (batch + 1) * batch_size]
        nrows = sample_ids.size // 7 if sample_ids.size % 7 == 0 else sample_ids.size // 7 + 1
        fig, ax = plt.subplots(nrows, 7, figsize=(15, 20))
        ax = ax.ravel()
        for i in range(sample_ids.shape[0]):
            scan_image, label = get_image(sample_ids[i], df=df, transform=transform, data_directory=data_directory)
            ax[i].imshow(np.transpose(scan_image, [1, 2, 0]))
            ax[i].set_title(f"{label_names[label]}")
            ax[i].set_axis_off()
        fig.savefig(DATA_CURATION_DIR / "excluded_cropped" / f"thumbnails_{batch}.png")
        plt.show()
        plt.close()
