from typing import Optional, Dict, Union, List, Sequence

import matplotlib.pyplot as plt
import torch
import numpy as np
from DataSelection.deep_learning.plots_tensorboard import get_histogram_plot
from torch.utils.tensorboard import SummaryWriter


class GradientAnalysis():
    """
    A class to analyse gradient of loss with respect to model parameters.
    Intra-class gradient directions are extracted and compared against the mean gradient direction to understand
    the pattern of samples with noisy labels.
    """

    def __init__(self,
                 target_class: int,
                 model: torch.nn.Module,
                 total_num_samples: int,
                 **sample_specific_args: Dict) -> None:
        """
        :param target_class: Index of the class to be used for gradient analysis.
        :param model: Model object whose parameters are utilised for the gradient computation.
        """
        self.target_class = target_class
        self.layer = model.densenet121.classifier if hasattr(model, 'densenet') else model.fc  # type: ignore
        self.total_num_samples = total_num_samples
        self.clear_label_noise_ids: np.ndarray = sample_specific_args['clear_noisy_case_ids']
        self.ambiguous_label_noise_ids: np.ndarray = sample_specific_args['ambiguous_noisy_case_ids']

        self.num_params = self.count_parameters(self.layer)
        self._initialise_grad_vars()

    def add_gradients(self, loss: torch.Tensor, labels: torch.Tensor, global_indices: torch.Tensor) -> None:
        """
        :param loss: per sample loss computed for mini-batch.
        :param labels: target labels of the samples in mini-batch.
        """
        intra_class_indices = np.where(labels.cpu().numpy() == self.target_class)[0].tolist()

        # Collect gradients wrt. to fc and get mean gradient
        for mini_batch_ind in intra_class_indices:
            params: Sequence[torch.Tensor] = self.layer.parameters()  # type: ignore
            _grads = torch.autograd.grad(loss[mini_batch_ind], (params), retain_graph=True)
            _grads_concat = torch.cat([_grads[ii].flatten() for ii in range(len(_grads))])

            global_ind = global_indices[mini_batch_ind]
            self.grads[global_ind, :] = _grads_concat
            self.sample_ids.append(global_ind)

    def log_epoch(self, epoch: int, writer: Optional[SummaryWriter]) -> None:
        """
        Creates a histogram plot of cosine similarities and writes in tensorboard events.
        """

        # Extract cosine similarity and re-order the indices
        sample_ids = np.array(self.sample_ids)
        cos = np.full(self.total_num_samples, np.nan)
        cos[sample_ids] = self._compute_cosine_similarity(self.grads[sample_ids, :]).cpu().numpy()

        # Identify subset of the dataset
        clear_noise_ids = np.intersect1d(sample_ids, self.clear_label_noise_ids)
        amb_noise_ids = np.intersect1d(sample_ids, self.ambiguous_label_noise_ids)
        all_noise_ids = np.union1d(clear_noise_ids, amb_noise_ids)
        no_noise_ids = np.setdiff1d(sample_ids, all_noise_ids)

        if writer:
            fig = get_histogram_plot(data=[cos[no_noise_ids], cos[clear_noise_ids], cos[amb_noise_ids]], num_bins=20,
                title=f"Histogram of cosine similarity between gradients - class {self.target_class}",
                x_label="Intra-class cosine similarity wrt to class mean",
                x_lim=(-1.0, 1.0))
            writer.add_figure('Histogram of intra-class cosine similarity', figure=fig, global_step=epoch, close=True)
            plt.close('all')

        # Clean the list of gradients
        self._initialise_grad_vars()

    def _compute_cosine_similarity(self, grads: torch.Tensor) -> torch.Tensor:
        """
        Cosine similarity metric of each samples gradient direction with respect to class mean.
        """

        # Normalise the gradients for angle computation
        grads_norm = torch.nn.functional.normalize(grads, dim=1, p=2)
        mean_grads_norm = torch.mean(grads_norm, dim=0)
        cos = torch.mv(grads_norm, mean_grads_norm)

        return cos

    def _initialise_grad_vars(self) -> None:
        self.grads = torch.zeros(self.total_num_samples, self.num_params, device='cuda:0')
        self.sample_ids: List[torch.Tensor] = list()

    @classmethod
    def count_parameters(self, model: Union[torch.Tensor, torch.nn.Module]) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)  # type: ignore
