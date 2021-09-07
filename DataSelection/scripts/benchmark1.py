#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


import argparse
from typing import Dict

import numpy as np

from DataSelection.selection.data_curation_utils import get_user_specified_selectors
from DataSelection.evaluation.plot_stats import plot_stats_scores
from DataSelection.selection.selectors.base import SampleSelector
from DataSelection.selection.simulation_statistics import get_ambiguous_sample_ids, get_mislabelled_sample_ids
from DataSelection.selection.selectors.label_based import LabelBasedDecisionRule, PosteriorBasedSelector
from DataSelection.utils.default_paths import BENCHMARK1_DIR
from DataSelection.utils.generic import create_folder, get_data_selection_parser, get_logger
from DataSelection.utils.dataset_utils import load_dataset_and_initial_labels_for_simulation


def main(args: argparse.Namespace) -> None:
    dataset, initial_labels = load_dataset_and_initial_labels_for_simulation(args.config[0], on_val_set=False)
    n_samples = dataset.num_samples
    true_distribution = dataset.label_distribution
    user_specified_selectors = get_user_specified_selectors(list_configs=args.config,
                                                            dataset=dataset,
                                                            output_path=BENCHMARK1_DIR,
                                                            plot_embeddings=args.plot_embeddings)
    # Data selection simulations for annotation
    default_selector: Dict[str, SampleSelector] = {
        'Oracle': PosteriorBasedSelector(dataset.label_distribution.distribution, n_samples,
                                         dataset.num_classes, name='Oracle',
                                         allow_repeat_samples=True,
                                         decision_rule=LabelBasedDecisionRule.TOTAL_VARIATION)}
    sample_selectors = {**default_selector, **user_specified_selectors}

    # Benchmark 1
    true_mislabelled = np.zeros(n_samples)
    true_mislabelled[get_mislabelled_sample_ids(initial_labels, true_distribution.label_counts)] = 1
    true_ambiguous = np.zeros(n_samples)
    true_ambiguous[get_ambiguous_sample_ids(true_distribution.label_counts)] = 1
    for name, sample_selector in sample_selectors.items():
        mislabelled_scores = sample_selector.get_mislabelled_scores(initial_labels)
        ambiguity_scores = sample_selector.get_ambiguity_scores(initial_labels)
        plot_stats_scores(sample_selector.name, mislabelled_scores, true_mislabelled, ambiguity_scores, true_ambiguous,
                          save_path=BENCHMARK1_DIR)


if __name__ == '__main__':
    create_folder(BENCHMARK1_DIR)
    get_logger(log_path=BENCHMARK1_DIR / 'dataread.log')
    parser = get_data_selection_parser()
    args, unknown_args = parser.parse_known_args()
    main(args)
