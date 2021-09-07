import logging
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import torch
from sklearn.metrics import auc, f1_score, precision_recall_curve, roc_auc_score
from torch.utils.tensorboard import SummaryWriter

@dataclass
class SampleMetrics():
    """
    Stores data required for training monitoring of individual model and
    post-training analysis in sample selection
    """
    name: str
    num_epochs: int
    num_samples: int
    num_classes: int
    ambiguous_mislabelled_ids: np.ndarray = None
    clear_mislabelled_ids: np.ndarray = None
    true_label_entropy: np.ndarray = None
    embeddings_size: Optional[int] = None

    def __post_init__(self) -> None:
        self.reset()
        self.loss_per_sample = np.full([self.num_samples, self.num_epochs], np.nan)
        self.logits_per_sample = np.full([self.num_samples, self.num_classes, self.num_epochs], np.nan)

        if self.clear_mislabelled_ids is not None:
            full_set = set(range(self.num_samples))
            if self.ambiguous_mislabelled_ids is not None:
                self.all_mislabelled_ids = np.concatenate([self.ambiguous_mislabelled_ids, self.clear_mislabelled_ids], 0)
            else:
                self.all_mislabelled_ids = self.clear_mislabelled_ids
            self.all_clean_label_ids = np.array(list(full_set.difference(self.all_mislabelled_ids)))

    def reset(self) -> None:
        self.loss_optimised: List[float] = list()
        self.predictions = np.full([self.num_samples], np.nan)
        self.labels = np.full([self.num_samples], np.nan)
        self.probabilities = np.full([self.num_samples, self.num_classes], np.nan)
        self.correct_predictions_teacher = np.full(self.num_samples, np.nan)
        self.correct_predictions = np.full([self.num_samples], np.nan)
        self.embeddings_per_sample = np.full([self.num_samples, self.embeddings_size],
                                             np.nan) if self.embeddings_size is not None else None

    def log_results(self, writer: SummaryWriter, epoch: int, name: str) -> None:
        mean_loss = np.mean(self.loss_optimised)
        accuracy = np.nanmean(self.correct_predictions)
        logging.info(f'{self.name} \t accuracy: {accuracy:.2f} \t loss: {mean_loss:.2e}')

        # Binary classification case
        if self.num_classes == 2:
            logits = self.logits_per_sample[:, :, epoch]
            labels = self.labels
            roc_auc = roc_auc_score(labels.reshape(-1), logits[:, 1].reshape(-1))
            f1 = f1_score(labels, np.argmax(logits, axis=-1))
            precision, recall, _ = precision_recall_curve(labels.reshape(-1), logits[:, 1].reshape(-1))
            pr_auc = auc(recall, precision)
            writer.add_scalar(tag='roc_auc', scalar_value=roc_auc, global_step=epoch)
            writer.add_scalar(tag='f1_score', scalar_value=f1, global_step=epoch)
            writer.add_scalar(tag='pr_auc', scalar_value=pr_auc, global_step=epoch)
            logging.info(f'{name} \t roc_auc: {roc_auc: .2f} \t pr_auc: {pr_auc: .2f} \t f1_score: {f1: .2f}')

        # Store accuracy and loss metrics
        writer.add_scalar(tag='loss', scalar_value=mean_loss, global_step=epoch)
        writer.add_scalar(tag='accuracy', scalar_value=accuracy, global_step=epoch)

        # Add histogram for the loss values
        self.log_loss_values(writer, self.loss_per_sample[:, epoch], epoch)

        # Breakdown of the accuracy on different sample types
        if self.clear_mislabelled_ids is not None:
            get_sub_acc = lambda ind: np.mean(self.correct_predictions[ind])
            scalar_dict = {
                'clean_cases': get_sub_acc(self.all_clean_label_ids),
                'all_mislabelled_cases': get_sub_acc(self.all_mislabelled_ids),
                'mislabelled_clear_cases': get_sub_acc(self.clear_mislabelled_ids)}
            if self.ambiguous_mislabelled_ids is not None:
                scalar_dict.update({'mislabelled_ambiguous_cases': get_sub_acc(self.ambiguous_mislabelled_ids)})
            writer.add_scalars(main_tag='accuracy_breakdown', tag_scalar_dict=scalar_dict, global_step=epoch)

        # Log mean teacher's accuracy
        if not np.isnan(self.correct_predictions_teacher).any():
            writer.add_scalar("teacher_accuracy", np.nanmean(self.correct_predictions_teacher), epoch)

    def get_margin(self, epoch: int) -> np.ndarray:
        """
        Get the margin for each sample defined as logits(y) - max_{y != t}[logits(t)]
        """
        margin = np.full(self.num_samples, np.nan)
        logits = self.logits_per_sample[:, :, epoch]
        for i in range(self.num_samples):
            label = int(self.labels[i])
            assigned_logit = logits[i, label]
            order = np.argsort(logits[i])
            order = order[order != label]
            other_max_logits = logits[i, order[-1]]
            margin[i] = assigned_logit - other_max_logits
        return margin

    def log_loss_values(self, writer: SummaryWriter, loss_values: np.ndarray, epoch: int) -> None:
        """
        Logs histogram of loss values of one of the co-teaching models.
        """
        writer.add_histogram('loss/all', loss_values, epoch)

    def append_batch(
            self,
            epoch: int,
            logits: torch.Tensor,
            labels: torch.Tensor,
            loss: float,
            indices: list,
            per_sample_loss: torch.Tensor,
            embeddings: Optional[np.ndarray] = None,
            teacher_logits: Optional[torch.Tensor] = None) -> None:
        """
        Append stats collected from batch of samples to metrics
        """
        if teacher_logits is not None:
            self.correct_predictions_teacher[indices] = torch.eq(torch.argmax(teacher_logits, dim=-1),
                                                                                labels).type(torch.float32).cpu()
        self.correct_predictions[indices] = torch.eq(torch.argmax(logits, dim=-1), labels).type(torch.float32).cpu()
        self.predictions[indices] = torch.argmax(logits, dim=-1).cpu()
        self.loss_per_sample[indices, epoch] = per_sample_loss.cpu()
        self.logits_per_sample[indices, :, epoch] = logits.cpu()
        self.probabilities[indices, :] = torch.softmax(logits, dim=-1).cpu()
        self.labels[indices] = labels.cpu()
        self.loss_optimised.append(loss)

        if embeddings is not None:
            # We don't know the size of the features in advance.
            if self.embeddings_size is None:
                self.embeddings_size = embeddings.shape[1]
                self.embeddings_per_sample = np.full([self.num_samples, self.embeddings_size], np.nan)
            assert self.embeddings_per_sample is not None
            self.embeddings_per_sample[indices, :] = embeddings
