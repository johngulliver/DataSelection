#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


import argparse
import copy
import multiprocessing
from typing import Tuple, Dict, Any
from itertools import product

import numpy as np
from joblib import Parallel, delayed

from DataSelection.evaluation.plot_stats import plot_stats
from DataSelection.selection.data_curation_utils import get_user_specified_selectors
from DataSelection.selection.selectors.base import SampleSelector
from DataSelection.selection.selectors.label_based import LabelBasedDecisionRule, PosteriorBasedSelector
from DataSelection.selection.selectors.random_selector import RandomSelector
from DataSelection.selection.simulation import DataCurationSimulator
from DataSelection.selection.simulation_statistics import SimulationStats, SimulationStatsDistribution
from DataSelection.utils.dataset_utils import load_dataset_and_initial_labels_for_simulation
from DataSelection.utils.default_paths import BENCHMARK2_DIR
from DataSelection.utils.generic import create_folder, get_data_selection_parser, get_logger


def main(args: argparse.Namespace) -> None:
    dataset, initial_labels = load_dataset_and_initial_labels_for_simulation(args.config[0], args.on_val_set)
    n_classes = dataset.num_classes
    n_samples = dataset.num_samples
    true_distribution = dataset.label_distribution

    user_specified_selectors = get_user_specified_selectors(list_configs=args.config,
                                                            dataset=dataset,
                                                            output_path=BENCHMARK2_DIR,
                                                            plot_embeddings=args.plot_embeddings)

    # Data selection simulations for annotation
    default_selector = {
        'Random': RandomSelector(n_samples, n_classes, name='Random'),
        'Oracle': PosteriorBasedSelector(true_distribution.distribution, n_samples, n_classes,
                                         name='Oracle',
                                         allow_repeat_samples=True,
                                         decision_rule=LabelBasedDecisionRule.INV)}
    sample_selectors = {**user_specified_selectors, **default_selector}

    # Benchmark 2
    # Determine the number of simulation iterations based on the noise rate
    expected_noise_rate = np.mean(np.argmax(true_distribution.distribution, -1) != dataset.targets[:n_samples])
    if dataset.name == "NoisyKaggle":
        relabel_budget = min(int(n_samples * expected_noise_rate * 3.0), n_samples)
    else:
        relabel_budget = min(int(n_samples * expected_noise_rate * 4.0), n_samples)

    # Setup the simulation function.
    def _run_simulation_for_selector(name: str, seed: int,
                                     sample_selector: SampleSelector) -> Tuple[str, SimulationStats]:
        simulator = DataCurationSimulator(initial_labels=copy.deepcopy(initial_labels),
                                          label_distribution=copy.deepcopy(true_distribution),
                                          relabel_budget=relabel_budget,
                                          name=name,
                                          seed=seed,
                                          sample_selector=copy.deepcopy(sample_selector))
        simulator.run_simulation()
        return name, simulator.global_stats

    # Run the simulation over multiple seeds and selectors
    simulation_iter = product(sample_selectors.items(), args.seeds)
    if args.debug:
        parallel_output = [_run_simulation_for_selector(_name, _seed, _sel) for (_name, _sel), _seed in simulation_iter]
    else:
        num_jobs = min(len(sample_selectors) * len(args.seeds), multiprocessing.cpu_count())
        parallel = Parallel(n_jobs=num_jobs)
        parallel_output = parallel(delayed(_run_simulation_for_selector)(_name, _seed, _selector)
            for (_name, _selector), _seed in simulation_iter)

    # Aggregate parallel output arrays
    global_stats: Dict[str, Any] = {name: list() for name in set([name for name, _ in parallel_output])}
    [global_stats[name].append(stats) for name, stats in parallel_output]

    # Analyse simulation stats
    stats_dist = {name: SimulationStatsDistribution(stats) for name, stats in global_stats.items()}
    plot_filename_suffix = "val" if args.on_val_set else "train"
    plot_stats(stats_dist,
               dataset_name=dataset.name,
               n_samples=n_samples,
               save_path=BENCHMARK2_DIR,
               filename_suffix=plot_filename_suffix)


if __name__ == '__main__':
    create_folder(BENCHMARK2_DIR)
    get_logger(BENCHMARK2_DIR / 'dataread.log')
    parser = get_data_selection_parser()
    args, unknown_args = parser.parse_known_args()
    main(args)
