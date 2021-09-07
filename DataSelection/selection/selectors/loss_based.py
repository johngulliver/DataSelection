#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import numpy as np
from DataSelection.selection.selectors.base import SampleSelector
from enum import Enum


class TrainingLossDynamicDecisionRule(Enum):
    MEAN = 1
    MEDIAN = 2
    MAX = 3
    MIN = 4
    STD = 5


class TrainingLossDynamicsBasedSelector(SampleSelector):
    def __init__(self,
                 loss_array: np.ndarray,
                 num_samples: int,
                 num_classes: int,
                 burnout_period_epochs: int = 0,
                 decision_rule: TrainingLossDynamicDecisionRule = TrainingLossDynamicDecisionRule.MEAN,
                 name: str = "Loss Dynamic Based Selector",
                 allow_repeat_samples: bool = False):
        super().__init__(num_samples, num_classes, name, allow_repeat_samples)
        assert loss_array.ndim == 2
        assert burnout_period_epochs >= 0
        self.loss_array = loss_array[:, burnout_period_epochs:]
        self.decision_rule_fn = getattr(np, decision_rule.name.lower())

    def get_relabelling_scores(self, current_labels: np.ndarray) -> np.ndarray:
        return self.decision_rule_fn(self.loss_array, axis=-1)

    def get_ambiguity_scores(self, current_labels: np.ndarray) -> np.ndarray:
        return self.decision_rule_fn(self.loss_array, axis=-1)

    def get_mislabelled_scores(self, current_labels: np.ndarray) -> np.ndarray:
        return self.decision_rule_fn(self.loss_array, axis=-1)
