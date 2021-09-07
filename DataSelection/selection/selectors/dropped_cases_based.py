#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import numpy as np
from DataSelection.selection.selectors.base import SampleSelector


class DroppedCasesDynamicsBasedSelector(SampleSelector):
    def __init__(self,
                 dropped_cases: np.ndarray,
                 num_samples: int,
                 num_classes: int,
                 burnout_period_epochs: int = 0,
                 name: str = "Dropped Cases Dynamic Based Selector",
                 allow_repeat_samples: bool = False):
        super().__init__(num_samples, num_classes, name, allow_repeat_samples)
        assert burnout_period_epochs >= 0
        self.dropped_cases = dropped_cases[:, burnout_period_epochs:]
        self.decision_rule_fn = np.mean

    def get_relabelling_scores(self, current_labels: np.ndarray) -> np.ndarray:
        return self.decision_rule_fn(self.dropped_cases, axis=-1)

    def get_ambiguity_scores(self, current_labels: np.ndarray) -> np.ndarray:
        return self.decision_rule_fn(self.dropped_cases, axis=-1)

    def get_mislabelled_scores(self, current_labels: np.ndarray) -> np.ndarray:
        return self.decision_rule_fn(self.dropped_cases, axis=-1)
