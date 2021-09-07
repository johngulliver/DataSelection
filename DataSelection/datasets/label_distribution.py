#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import numpy as np

class LabelDistribution(object):
    """
    LabelDistribution class handles sampling from a label distribution with reproducible behavior given a seed
    """

    def __init__(self, seed: int, label_counts: np.ndarray) -> None:
        """

        :param seed: The random seed used to ensure reproducible behaviour
        :param label_counts: An array of shape (num_samples, num_classes) where each entry represents the number of
        labels available for each sample and class
        """
        assert label_counts.dtype == np.int64
        assert label_counts.ndim == 2
        self.num_classes = label_counts.shape[1]
        self.num_samples = label_counts.shape[0]
        self.seed = seed
        self.random_state = np.random.RandomState(seed)
        self.label_counts = label_counts

        # make the distribution
        self.distribution = label_counts / np.sum(label_counts, axis=1, keepdims=True)
        assert np.isfinite(self.distribution).all()

    def sample_initial_labels_for_all(self, temperature: float = 1.0) -> np.ndarray:
        """
        Sample one label for each sample in the dataset according to its label distribution
        :param temperature: A temperature a value that will be used to temperature scale the distribution, default is
        1.0  which is equivalent to no scaling; temperature must be greater than 0.0, values between 0 and 1 will result
        in a sharper distribution and values greater than 1 in a more uniform distribution over classes
        :return: None
        """
        assert temperature > 0
        distribution = np.power(self.distribution, 1. / temperature)
        distribution = distribution / np.sum(distribution, axis=1, keepdims=True)
        sampling_fn = lambda p: self.random_state.choice(self.num_classes, 1, p=p)

        return np.squeeze(np.apply_along_axis(sampling_fn, arr=distribution, axis=1))

    def sample(self, sample_idx: int) -> int:
        """
        Sample one label for a given sample index
        :param sample_idx: The sample index for which the label will be sampled
        :return: None
        """
        return self.random_state.choice(self.num_classes, 1, p=self.distribution[sample_idx])[0]
