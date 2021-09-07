#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import logging
import numpy as np

from DataSelection.evaluation.plot_stats import plot_relabeling_score
from DataSelection.selection.selectors.base import SampleSelector
from DataSelection.selection.selectors.graph import GraphBasedSelector
from DataSelection.selection.simulation_statistics import SimulationStats
from DataSelection.datasets.label_distribution import LabelDistribution


class DataCurationSimulator(object):
    """
    Class that runs the reactive learning simulation given a Selector object
    """

    def __init__(self,
                 initial_labels: np.ndarray,
                 label_distribution: LabelDistribution,
                 relabel_budget: int,
                 sample_selector: SampleSelector,
                 seed: int = 1234,
                 name: str = "Default Simulation") -> None:
        super().__init__()
        """
        """
        self.relabel_budget = relabel_budget
        self.name = name
        self.random_seed = seed
        self.sample_selector = sample_selector
        label_distribution.random_state = np.random.RandomState(seed)

        # Initialise the label pool and first set of labels for the simulation
        self._current_labels = initial_labels
        self._label_distribution = label_distribution
        self._global_stats = SimulationStats(name, label_distribution.label_counts, initial_labels)
        plot_relabeling_score(true_majority=np.argmax(self._label_distribution.label_counts, axis=1),
                              starting_scores=self.sample_selector.get_relabelling_scores(self._current_labels),
                              current_majority=np.argmax(self._current_labels, axis=1),
                              selector_name=self.sample_selector.name)

    def fetch_until_majority(self, sample_idx: int) -> int:
        """
        Sample labels unitl a majority is formed for a given sample index
        :param sample_idx: The sample index for which the labels will be sampled
        :return:
        """
        majority_formed = False
        num_fetches_per_sample = 0
        while not majority_formed:
            label = self._label_distribution.sample(sample_idx)
            self._current_labels[sample_idx, label] += 1
            _arr = np.sort(self._current_labels[sample_idx])
            majority_formed = _arr[-1] != _arr[-2]
            num_fetches_per_sample += 1
            logging.debug(f"Sample ID: {sample_idx}, Selected label: {label}")
            logging.debug(f"Sample ID: {sample_idx}, Current labels: {self._current_labels[sample_idx]}")

        return num_fetches_per_sample

    def run_simulation(self, plot_samples: bool = False) -> None:
        """
        """
        logging.info(f"Running Simulation Using {self.sample_selector.name} Selector ...")
        num_relabels = 0
        _iter = 0
        while num_relabels <= self.relabel_budget:
            logging.info(f"\nIteration {_iter}")
            sample_id = int(self.sample_selector.get_batch_of_samples_to_annotate(self._current_labels, 1)[0])
            num_fetches = self.fetch_until_majority(sample_id)

            self._global_stats.record_iteration(sample_id, num_fetches, self._current_labels)
            self._global_stats.log_last_iter()
            num_relabels += num_fetches
            _iter += 1
            if isinstance(self.sample_selector, GraphBasedSelector) and plot_samples:
                self.sample_selector.plot_selected_sample(sample_id, include_knn=False)

    @property
    def global_stats(self) -> SimulationStats:
        return self._global_stats

    @property
    def current_labels(self) -> np.ndarray:
        return self._current_labels
