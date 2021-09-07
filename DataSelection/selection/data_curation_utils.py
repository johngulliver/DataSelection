#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score

from DataSelection.configs.config_node import ConfigNode
from DataSelection.deep_learning.model_inference import inference_ensemble
from DataSelection.deep_learning.utils import load_selector_config
from DataSelection.selection.selectors.base import SampleSelector
from DataSelection.selection.selectors.dropped_cases_based import DroppedCasesDynamicsBasedSelector
from DataSelection.selection.selectors.graph import GraphBasedSelector, GraphParameters
from DataSelection.selection.selectors.label_based import LabelBasedDecisionRule, PosteriorBasedSelector
from DataSelection.selection.selectors.loss_based import TrainingLossDynamicsBasedSelector
from DataSelection.selection.simulation_statistics import get_ambiguous_sample_ids
from DataSelection.utils.custom_types import SelectorTypes as ST
from DataSelection.utils.default_paths import get_train_output_dir
from DataSelection.utils.plot import plot_model_embeddings


def evaluate_ambiguous_case_detection(bald_score: np.ndarray, labels_complete: np.ndarray) -> None:
    uncertain_cases = np.zeros_like(bald_score)
    true_ambiguous_cases = get_ambiguous_sample_ids(labels_complete)
    uncertain_cases[true_ambiguous_cases] = 1
    auc_ = roc_auc_score(y_true=uncertain_cases, y_score=bald_score)
    logging.info(f'BALD ambiguous detection AUC: {float(auc_):.2f}')


def get_inference_outputs_per_model(embeddings: np.ndarray,
                                    posteriors: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param embeddings: 4D numpy array - containing sample embeddings obtained from a CNN.
                       [num_ensemble_runs, num_models, num_samples, embedding_size]
    :param posteriors: 4D numpy array - containing class posteriors obtained from a CNN.
                       [num_ensemble_runs, num_models, num_samples, num_classes]
    """
    # flatten embeddings along the ensemble and num_models dimension, average posteriors
    embeddings = embeddings[0, 0]
    posteriors = np.mean(posteriors, axis=(0, 1))
    return embeddings, posteriors


def pretty_selector_name(_type: str, model_name: str) -> str:
    type_dict = {'PosteriorBasedSelector': 'Posterior',
                 'PosteriorBasedSelectorInv': 'Posterior (inv)',
                 'TrainingLossDynamicsBasedSelector': 'Loss',
                 'GraphBasedSelector': 'Graph',
                 'DroppedCasesDynamicsBasedSelector': "Dropped Cases"}
    return f'{model_name} ({type_dict[_type]})'


def get_selector(_type: str, cfg: ConfigNode, **pars: Any) -> SampleSelector:
    num_samples = pars["num_samples"]
    num_classes = pars["num_classes"]
    path_train_metrics = pars["path_train_metrics"]
    sample_indices = pars["sample_indices"]
    name = pars["name"]
    embeddings = pars["embeddings"]
    posteriors = pars["posteriors"]

    if ST(_type) is ST.GraphBasedSelector:
        distance_metric = "cosine" if (
                cfg.model.resnet.apply_l2_norm or cfg.train.use_self_supervision) else "euclidean"
        graph_params = GraphParameters(n_neighbors=num_samples // 200,
                                       diffusion_alpha=0.95,
                                       cg_solver_max_iter=10,
                                       diffusion_batch_size=num_samples // 100,
                                       distance_kernel=distance_metric)
        return GraphBasedSelector(num_samples, num_classes, embeddings,
                                  sample_indices=sample_indices, name=name,
                                  graph_params=graph_params)
    elif ST(_type) is ST.PosteriorBasedSelector:
        return PosteriorBasedSelector(posteriors, num_samples, num_classes, name=name)
    elif ST(_type) is ST.PosteriorBasedSelectorInv:
        return PosteriorBasedSelector(posteriors, num_samples, num_classes, name=name,
                                      decision_rule=LabelBasedDecisionRule.INV)
    elif ST(_type) is ST.TrainingLossDynamicsBasedSelector:
        # Load arrays for training dynamics
        training_metrics = np.load(path_train_metrics)
        loss_array = training_metrics['loss_per_sample'][:num_samples]
        return TrainingLossDynamicsBasedSelector(loss_array, num_samples, num_classes, name=name,
                                                 burnout_period_epochs=cfg.selector.burnout_period)
    elif ST(_type) is ST.DroppedCasesDynamicsBasedSelector:
        # Load arrays for training dynamics
        training_metrics = np.load(path_train_metrics)
        dropped_cases = training_metrics['dropped_cases'][:num_samples]
        return DroppedCasesDynamicsBasedSelector(dropped_cases, num_samples, num_classes, name=name,
                                                 burnout_period_epochs=cfg.selector.burnout_period)
    else:
        raise ValueError("Unknown selector type is specified")


def get_user_specified_selectors(list_configs: List[str],
                                 dataset: Any,
                                 output_path: Path,
                                 plot_embeddings: bool = False) -> Dict[str, SampleSelector]:
    """
    Load the user specific configs, get the embeddings and return the selectors.
    :param list_configs:
    :return: dictionary of selector
    """
    logging.info("Loading the selector configs:\n {0}".format('\n'.join(list_configs)))
    user_specified_selectors = dict()
    for cfg in [load_selector_config(cfg) for cfg in list_configs]:
        # Collect model probability predictions for the given set of images in the training set.
        embeddings, posteriors = inference_ensemble(dataset, cfg, 1)
        embeddings, posteriors = get_inference_outputs_per_model(embeddings, posteriors)
        path_train_metrics = Path(get_train_output_dir(cfg)) / "model_0_train" / "model_0_train_training_stats.npz"
        assert posteriors.shape[0] == dataset.num_samples

        if plot_embeddings:
            sample_label_counts = dataset.label_counts
            plot_model_embeddings(embeddings=embeddings, label_distribution=sample_label_counts,
                                  label_names=dataset.get_label_names(), save_path=output_path)

        for _type in cfg.selector.type:
            selector_params = {"path_train_metrics": path_train_metrics,
                               "num_classes": dataset.num_classes,
                               "num_samples": dataset.num_samples,
                               "sample_indices": dataset.indices,
                               "embeddings": embeddings,
                               "posteriors": posteriors,
                               "name": pretty_selector_name(_type, cfg.selector.model_name)}
            selector_name = pretty_selector_name(_type, cfg.selector.model_name)
            user_specified_selectors[selector_name] = get_selector(_type, cfg, **selector_params)

    return user_specified_selectors
