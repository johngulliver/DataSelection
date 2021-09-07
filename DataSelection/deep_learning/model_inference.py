#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
from typing import Any, List, Tuple

import numpy as np
from DataSelection.configs.config_node import ConfigNode
from DataSelection.deep_learning.co_teaching_trainer import CoTeachingTrainer
from DataSelection.deep_learning.collect_embeddings import get_all_embeddings, register_embeddings_collector
from DataSelection.deep_learning.utils import get_run_config
from DataSelection.deep_learning.vanilla_trainer import VanillaTrainer


def inference_single_model(dataset: Any, config: ConfigNode) -> Tuple[List, List]:
    """
    Performs an inference pass on a single model
    :param config:
    :return:
    """
    model_trainer_class = CoTeachingTrainer if config.train.use_co_teaching else VanillaTrainer
    model_trainer = model_trainer_class(config)
    model_trainer.load_checkpoints()

    all_model_cnn_embeddings = register_embeddings_collector(model_trainer.models, use_only_in_train=False)

    # Inference on the Training Set
    trackers = model_trainer.run_inference(dataset)
    embs = get_all_embeddings(all_model_cnn_embeddings)
    probs = [metric_tracker.sample_metrics.probabilities for metric_tracker in trackers]
    return embs, probs


def inference_ensemble(dataset: Any, config: ConfigNode, num_runs: int) -> Tuple[np.ndarray, np.ndarray]:
    all_embeddings = []
    all_posteriors = []
    for i, _ in enumerate(range(num_runs)):
        config_run = get_run_config(config, config.train.seed + i)
        embeddings, posteriors = inference_single_model(dataset, config_run)
        all_embeddings.append(embeddings)
        all_posteriors.append(posteriors)
    return np.array(all_embeddings), np.array(all_posteriors)
