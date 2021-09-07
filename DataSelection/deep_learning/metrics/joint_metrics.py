from dataclasses import dataclass
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from DataSelection.datasets.cifar10 import get_cifar10_label_names
from DataSelection.deep_learning.metrics.sample_metrics import SampleMetrics
from DataSelection.deep_learning.plots_tensorboard import (get_scatter_plot, plot_disagreement_per_sample,
                                                                 plot_excluded_cases_coteaching)
from DataSelection.deep_learning.transforms import ToNumpy
from DataSelection.utils.plot import plot_latent_space_and_noise_cases
from sklearn import mixture
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10


@dataclass()
class JointMetrics():
    """
    Stores metrics for co-teaching models.
    """
    num_samples: int
    num_epochs: int
    dataset: Optional[Any] = None
    ambiguous_mislabelled_ids: np.ndarray = None
    clear_mislabelled_ids: np.ndarray = None
    true_label_entropy: np.ndarray = None
    plot_class_margin_difference: bool = False

    def reset(self) -> None:
        self.kl_divergence_symmetric = np.full([self.num_samples], np.nan)
        self.active = False

    def __post_init__(self) -> None:
        self.reset()
        self.prediction_disagreement = np.zeros([self.num_samples, self.num_epochs], dtype=np.bool)
        self._initialise_dataset_properties()
        self.case_drop_histogram = np.zeros([self.num_samples, self.num_epochs + 1], dtype=np.bool)
        if not isinstance(self.clear_mislabelled_ids, np.ndarray) or not isinstance(self.ambiguous_mislabelled_ids,
                                                                                    np.ndarray):
            return

        self.true_mislabelled_ids = np.concatenate([self.ambiguous_mislabelled_ids, self.clear_mislabelled_ids], axis=0)
        self.case_drop_histogram[self.clear_mislabelled_ids, -1] = True
        self.case_drop_histogram[self.ambiguous_mislabelled_ids, -1] = True

    def _initialise_dataset_properties(self) -> None:
        if self.dataset is not None:
            self.label_names = get_cifar10_label_names() if isinstance(self.dataset, CIFAR10) \
                else self.dataset.get_label_names()
            self.dataset.transform = ToNumpy()  # type: ignore

    def log_results(self, writer: SummaryWriter, epoch: int, sample_metrics: SampleMetrics) -> None:
        if (not self.active) or (self.ambiguous_mislabelled_ids is None) or (self.clear_mislabelled_ids is None):
            return

        # KL Divergence between the two posteriors
        writer.add_scalars(main_tag='symmetric-kl-divergence', tag_scalar_dict={
            'all': np.nanmean(self.kl_divergence_symmetric),
            'ambiguous': np.nanmean(self.kl_divergence_symmetric[self.ambiguous_mislabelled_ids]),
            'clear_noise': np.nanmean(self.kl_divergence_symmetric[self.clear_mislabelled_ids])},
                           global_step=epoch)

        # Disagreement rate between the models
        writer.add_scalars(main_tag='disagreement_rate', tag_scalar_dict={
            'all': np.nanmean(self.prediction_disagreement[:, epoch]),
            'ambiguous': np.nanmean(self.prediction_disagreement[self.ambiguous_mislabelled_ids, epoch]),
            'clear_noise': np.nanmean(self.prediction_disagreement[self.clear_mislabelled_ids, epoch])},
                           global_step=epoch)

        # Add histogram for the loss values
        self.log_loss_values(writer, sample_metrics.loss_per_sample[:, epoch], epoch)

        # Add disagreement metrics
        fig = get_scatter_plot(self.true_label_entropy, self.kl_divergence_symmetric,
                               x_label="Label entropy", y_label="Symmetric-KL", y_lim=[0.0, 2.0])
        writer.add_figure('Sym-KL vs Label Entropy', figure=fig, global_step=epoch, close=True)

        fig = plot_disagreement_per_sample(self.prediction_disagreement, self.true_label_entropy)
        writer.add_figure('Disagreement of prediction', figure=fig, global_step=epoch, close=True)

        # Excluded cases diagnostics
        self.log_dropped_cases_metrics(writer=writer, epoch=epoch)

        # Every 10 epochs, display the dropped cases in the co-teaching algorithm
        if epoch % 10:
            self.log_dropped_images(writer=writer, predictions=sample_metrics.predictions, epoch=epoch)

        # Every 100 epochs logs the embeddings and the dropped cases
        if epoch > 0 and epoch % 100 == 0 and sample_metrics.embeddings_per_sample is not None:
            self.log_embeddings(writer=writer,
                                labels=sample_metrics.labels,
                                embeddings=sample_metrics.embeddings_per_sample,
                                epoch=epoch)

        # Histogram of the margins
        if self.plot_class_margin_difference:
            self.log_margin_histograms(writer=writer, sample_metrics=sample_metrics, epoch=epoch)

        # Close all figures
        plt.close('all')

    def log_dropped_cases_metrics(self, writer: SummaryWriter, epoch: int) -> None:
        """
        Creates all diagnostics for dropped cases analysis.
        """
        entropy_sorted_indices = np.argsort(self.true_label_entropy)
        drop_cur_epoch_mask = self.case_drop_histogram[:, epoch]
        drop_cur_epoch_ids = np.where(drop_cur_epoch_mask)[0]
        is_sample_dropped = np.any(drop_cur_epoch_mask)
        title = None
        if is_sample_dropped:
            n_dropped = float(drop_cur_epoch_ids.size)
            average_label_entropy_dropped_cases = np.mean(self.true_label_entropy[drop_cur_epoch_mask])
            n_detected_mislabelled = np.intersect1d(drop_cur_epoch_ids, self.true_mislabelled_ids).size
            n_clean_dropped = int(n_dropped - n_detected_mislabelled)
            n_detected_mislabelled_ambiguous = np.intersect1d(drop_cur_epoch_ids, self.ambiguous_mislabelled_ids).size
            n_detected_mislabelled_clear = np.intersect1d(drop_cur_epoch_ids, self.clear_mislabelled_ids).size
            perc_detected_mislabelled = n_detected_mislabelled / n_dropped * 100
            perc_detected_clear_mislabelled = n_detected_mislabelled_clear / n_dropped * 100
            perc_detected_ambiguous_mislabelled = n_detected_mislabelled_ambiguous / n_dropped * 100
            title = f"Dropped Cases: Avg label entropy {average_label_entropy_dropped_cases:.3f}\n " \
                    f"Dropped cases: {n_detected_mislabelled} mislabelled ({perc_detected_mislabelled:.1f}%) - " \
                    f"{n_clean_dropped} clean ({(100 - perc_detected_mislabelled):.1f}%)\n" \
                    f"Num ambiguous mislabelled among detected cases: {n_detected_mislabelled_ambiguous}" \
                    f" ({perc_detected_ambiguous_mislabelled:.1f}%)\n" \
                    f"Num clear mislabelled among detected cases: {n_detected_mislabelled_clear}" \
                    f" ({perc_detected_clear_mislabelled:.1f}%)"
            writer.add_scalars(main_tag='Number of dropped cases', tag_scalar_dict={
                'clean_cases': n_clean_dropped,
                'all_mislabelled_cases': n_detected_mislabelled,
                'mislabelled_clear_cases': n_detected_mislabelled_clear,
                'mislabelled_ambiguous_cases': n_detected_mislabelled_ambiguous}, global_step=epoch)
            writer.add_scalar(tag="Percentage of mislabelled among dropped cases",
                              scalar_value=perc_detected_mislabelled, global_step=epoch)
        fig = plot_excluded_cases_coteaching(case_drop_mask=self.case_drop_histogram,
                                             entropy_sorted_indices=entropy_sorted_indices, title=title,
                                             num_epochs=self.num_epochs, num_samples=self.num_samples)
        writer.add_figure('Histogram of excluded cases', figure=fig, global_step=epoch, close=True)

    def log_loss_values(self, writer: SummaryWriter, loss_values: np.ndarray, epoch: int) -> None:
        """
        Logs histogram of loss values of one of the co-teaching models.
        """
        writer.add_histogram('loss/all', loss_values, epoch)
        writer.add_histogram('loss/ambiguous_noise', loss_values[self.ambiguous_mislabelled_ids], epoch)
        writer.add_histogram('loss/clear_noise', loss_values[self.clear_mislabelled_ids], epoch)

    def log_embeddings(self, writer: SummaryWriter, labels: np.ndarray, embeddings: np.ndarray, epoch: int) -> None:
        """
        Logs embeddings per class, indicating noisy and dropped cases.
        """
        labels = [self.label_names[int(x)] for x in labels.tolist()]
        metadata = ["Mislabelled" if i in self.true_mislabelled_ids else "Clean" for i in range(self.num_samples)]
        figure, ax = plt.subplots(figsize=(14, 12))
        plot_latent_space_and_noise_cases(embeddings=embeddings,
                                          labels=np.array(labels),
                                          indicator_noisy_labels=np.asarray(metadata),
                                          selected_cases=self.case_drop_histogram[:, epoch],
                                          ax=ax)
        writer.add_figure(tag="Embeddings", figure=figure, global_step=epoch)

    def log_dropped_images(self, writer: SummaryWriter, predictions: np.ndarray, epoch: int) -> None:
        """
        Logs images dropped during co-teaching training
        """
        dropped_cases = np.where(self.case_drop_histogram[:, epoch])[0]
        if dropped_cases.size > 0 and self.dataset is not None:
            dropped_cases = dropped_cases[np.argsort(self.true_label_entropy[dropped_cases])]
            fig = self.plot_batch_images_and_labels(predictions, list_indices=dropped_cases[:64])
            writer.add_figure("Dropped images with lowest entropy", figure=fig, global_step=epoch, close=True)
            fig = self.plot_batch_images_and_labels(predictions, list_indices=dropped_cases[-64:])
            writer.add_figure("Dropped images with highest entropy", figure=fig, global_step=epoch, close=True)

            kept_cases = np.where(~self.case_drop_histogram[:, epoch])[0]
            kept_cases = kept_cases[np.argsort(self.true_label_entropy[kept_cases])]
            fig = self.plot_batch_images_and_labels(predictions, kept_cases[-64:])
            writer.add_figure("Kept images with highest entropy", figure=fig, global_step=epoch, close=True)

    def plot_batch_images_and_labels(self, predictions: np.ndarray, list_indices: np.ndarray) -> plt.Figure:
        """
        Plots of batch of images along with their labels and predictions. Noise cases are colored in red, clean cases
        in green. Images are assumed to be numpy images (use ToNumpy() transform).
        """
        assert self.dataset is not None
        fig, ax = plt.subplots(8, 8, figsize=(8, 10))
        ax = ax.ravel()
        for i, index in enumerate(list_indices):
            predicted = int(predictions[index])
            color = "red" if index in self.true_mislabelled_ids else "green"
            _, img, training_label = self.dataset.__getitem__(index)
            ax[i].imshow(img)
            ax[i].set_axis_off()
            ax[i].set_title(f"Label: {self.label_names[training_label]}\nPred: {self.label_names[predicted]}",
                            color=color, fontsize="x-small")
        return fig

    def log_margin_histograms(self, writer: SummaryWriter, sample_metrics: SampleMetrics, epoch: int) -> None:
        """
        Plot histogram of AUM for clean and mislabelled points
        """
        margin = sample_metrics.get_margin(epoch).reshape(-1, 1)
        clf = mixture.GaussianMixture(n_components=2, covariance_type="full")
        clf.fit(margin)
        x = np.linspace(margin.min(), margin.max(), 300).reshape(-1, 1)
        score = clf.score_samples(x)
        pdf = np.exp(score)
        responsibilities = clf.predict_proba(x.reshape(-1, 1))
        pdf_individual = responsibilities * pdf[:, np.newaxis]
        type_of_label = ["Mislabelled clear" if i in self.clear_mislabelled_ids
                         else "Mislabelled ambiguous" if i in self.ambiguous_mislabelled_ids
        else "Clean" for i in range(self.num_samples)]
        is_noise = ["Mislabelled" if i in self.true_mislabelled_ids else "Clean"
                    for i in range(self.num_samples)]
        df = pd.DataFrame({"margin": margin.reshape(-1), "type_of_label": type_of_label, "is_noise": is_noise})
        fig, ax = plt.subplots(1, 3, figsize=(18, 5))
        sns.histplot(data=df, x="margin", hue="is_noise", ax=ax[0], stat="density", common_norm=True)
        sns.histplot(data=df, x="margin", hue="is_noise", ax=ax[1], stat="probability", common_norm=False)
        sns.histplot(data=df, x="margin", hue="type_of_label", ax=ax[2], stat="probability", common_norm=False)
        ax[0].plot(x.reshape(-1), pdf, "-k")
        ax[0].plot(x, pdf_individual, '--k')
        ax[0].set_title("Histogram of margin, normalized overall")
        ax[1].set_title("Marginal histogram of margin (normalized by noise category)")
        writer.add_figure(tag="logits_margin", figure=fig, global_step=epoch)
