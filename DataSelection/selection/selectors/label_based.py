#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import logging
from enum import Enum
from typing import Optional

import numpy as np

from DataSelection.evaluation.metrics import cross_entropy, max_prediction_error, total_variation
from DataSelection.selection.selectors.base import SampleSelector


class LabelBasedDecisionRule(Enum):
    CROSS_ENTROPY = 1
    MAX_PREDICTION_ERROR = 2
    TOTAL_VARIATION = 3
    INV = 4


class LabelDistributionBasedSampler(SampleSelector):
    def __init__(self,
                 num_samples: int,
                 num_classes: int,
                 decision_rule: LabelBasedDecisionRule,
                 name: str = "Label Based Selector",
                 allow_repeat_samples: bool = False,
                 embeddings: Optional[np.ndarray] = None):
        super().__init__(num_samples=num_samples,
                         num_classes=num_classes,
                         name=name,
                         allow_repeat_samples=allow_repeat_samples,
                         embeddings=embeddings)
        self.decision_rule = decision_rule
        if decision_rule == LabelBasedDecisionRule.CROSS_ENTROPY:
            self.decision_fn = cross_entropy
        elif decision_rule in [LabelBasedDecisionRule.MAX_PREDICTION_ERROR, LabelBasedDecisionRule.INV]:
            self.decision_fn = max_prediction_error
        elif decision_rule == LabelBasedDecisionRule.TOTAL_VARIATION:
            self.decision_fn = total_variation
        else:
            raise ValueError('Invalid decision rule!')

    def get_predicted_label_distribution(self, current_labels: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_relabelling_scores(self, current_labels: np.ndarray) -> np.ndarray:
        predicted_label_distribution = self.get_predicted_label_distribution(current_labels)
        current_distribution = current_labels / np.sum(current_labels, axis=-1, keepdims=True)
        return self.decision_fn(predicted_label_distribution, current_distribution)

    def get_ambiguity_scores(self, current_labels: np.ndarray) -> np.ndarray:
        predicted_distribution = self.get_predicted_label_distribution(current_labels)
        return cross_entropy(predicted_distribution, predicted_distribution)

    def get_mislabelled_scores(self, current_labels: np.ndarray) -> np.array:
        current_distribution = current_labels / np.sum(current_labels, axis=-1, keepdims=True)
        return max_prediction_error(self.get_predicted_label_distribution(current_labels), current_distribution)


class PosteriorBasedSelector(LabelDistributionBasedSampler):
    """
    Selects samples based on the label posteriors computed by a model
    """

    def __init__(self,
                 predicted_label_distribution: np.ndarray,
                 num_samples: int,
                 num_classes: int,
                 embeddings: Optional[np.ndarray] = None,
                 name: str = "Posterior Selector",
                 decision_rule: LabelBasedDecisionRule = LabelBasedDecisionRule.MAX_PREDICTION_ERROR,
                 allow_repeat_samples: bool = False):
        super().__init__(num_samples=num_samples,
                         num_classes=num_classes,
                         decision_rule=decision_rule,
                         name=name,
                         allow_repeat_samples=allow_repeat_samples,
                         embeddings=embeddings)
        assert (len(predicted_label_distribution) == self.num_samples)
        self.predicted_label_distribution = predicted_label_distribution
        self.decision_rule = decision_rule
        self.exp_num_relabels = np.zeros(predicted_label_distribution.shape[0])
        self.current_labels = np.zeros_like(predicted_label_distribution)

    def get_predicted_label_distribution(self, current_labels: np.ndarray) -> np.ndarray:
        return self.predicted_label_distribution

    def get_relabelling_scores(self, current_labels: np.ndarray) -> np.ndarray:
        if self.decision_rule == LabelBasedDecisionRule.INV:
            relabel_scores = self.get_exp_relabel_scores(current_labels)
            current_distribution = current_labels / np.sum(current_labels, axis=-1, keepdims=True)
            mislabelled_scores = self.decision_fn(self.predicted_label_distribution, current_distribution)
            return mislabelled_scores / relabel_scores
        else:
            current_distribution = current_labels / np.sum(current_labels, axis=-1, keepdims=True)
            return self.decision_fn(self.predicted_label_distribution, current_distribution)

    def get_exp_relabel_scores(self, current_labels: np.ndarray) -> np.ndarray:
        all_classes = np.arange(current_labels.shape[1])
        majority_label = np.argmax(current_labels, axis=1)
        # Only compute if necessary i.e. if current_labels is different from last
        # round for a given sample
        rows = np.unique(np.where(current_labels != self.current_labels)[0])
        logging.info(f"Update {rows} subjects")
        for ix in rows:
            # get the chosen label and the predicted distribution for each image
            chosen = majority_label[ix]
            dist = self.predicted_label_distribution[ix, :]

            # if the distribution has a unit mass, use this info and don't use recursion
            if np.max(dist) == 1:
                self.exp_num_relabels[ix] = 1 if dist[chosen] == 1 else 2
            else:
                # then get the expected number of relabels
                all_classes_curr = all_classes[~np.isin(all_classes, chosen)]
                exp_relabel = 1 + self.exp_num_fun3(all_classes_curr, dist, ctr=0, max_levels=3)
                self.exp_num_relabels[ix] = exp_relabel
        self.current_labels = current_labels.copy()
        return self.exp_num_relabels

    def exp_num_fun3(self, remaining_classes: np.ndarray, dist: np.ndarray, ctr: int, max_levels: int = 2) -> float:
        # keep track of how many recursive calls were made
        ctr = ctr + 1
        if len(remaining_classes) == 0:
            return 0
        else:
            if ctr >= max_levels:  # if you reached the max depth in recursion
                return float(np.sum(dist[remaining_classes]))
            else:  # if you have not reached max depth yet
                sumval = 0
                for j in remaining_classes:
                    rem_classes_curr = remaining_classes[~np.isin(remaining_classes, j)]
                    sumval += dist[j] * (1 + self.exp_num_fun3(rem_classes_curr, dist, ctr, max_levels))
                return sumval
