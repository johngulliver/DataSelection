from typing import Optional, List, Any

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.text import Annotation
from matplotlib.widgets import Button


def plot_latent_space(embeddings: np.ndarray,
                      labels: np.ndarray,
                      markers: Optional[dict] = None,
                      ax: Optional[plt.axes] = None) -> None:
    """
    The default static plotting function to display data embeddings.

    :param embeddings: Input embeddings or latent space representations (num_points x feature dimension)
    :param labels: Target labels representing the embedding samples (num_points) - Only binary labels are supported
    :param markers: Markers representing easy, difficult, and noisy cases, e.g. {'easy': (data_index, 'orange')}
    :param ax: Axis of existing figure object, it is required interactive plotting.
    """

    # Visualise two dimensional embedding - perform k-mean clustering for visualisation.
    if ax is None:
        fig, ax = plt.subplots(figsize=(15, 12))

    def label_to_legend(x: int) -> str:
        if x == 1:
            return "positives"
        elif x == 0:
            return "negatives"
        else:
            return "label unavailable"

    data = pd.DataFrame({"emb1": embeddings[:, 0],
                         "emb2": embeddings[:, 1],
                         "class_type": list(map(label_to_legend, labels))})
    palette = [sns.color_palette("deep", 10)[0], sns.color_palette("deep", 10)[2]]
    sns.scatterplot(data=data, x="emb1", y="emb2", hue="class_type", s=45, alpha=0.75,
                    linewidths=.1, ax=ax, picker=True, palette=palette)
    ax.set_ylabel('Second component')
    ax.set_xlabel('First component')
    ax.set_title(f'Sample Categorisation - N={embeddings.shape[0]}')

    # Visualise the clear label noise and difficult cases
    if markers is not None:
        for label, (indices, color) in markers.items():
            ax.scatter(embeddings[indices, 0], embeddings[indices, 1],
                       c=color, s=30, label=label, marker='x', alpha=1, linewidths=0.5)

    ax.legend(loc='lower right')


def plot_latent_interactive(embeddings: np.ndarray,
                            labels: np.ndarray,
                            subject_ids: Optional[List[str]] = None,
                            report_df: Optional[pd.DataFrame] = None,
                            markers: dict = None) -> None:
    """
    Interactive plotting tool to visalise latent space.

    :param embeddings: Input embeddings or latent space representations (num_points x feature dimension)
    :param labels: Target labels representing the embedding samples (num_points) - Only binary labels are supported
    :param subject_ids: A list containing subject ID of each embedding sample point - This information is displayed
                        on interactive plot if user clicks on a data point
    :param report_df: A dataframe containing data input and model output information. This information is parsed to
                      display subject information in interactive plot
    :param markers: Markers representing easy, difficult, and noisy cases, e.g. {'easy': (data_index, 'orange')}
    """

    plot_kwargs = {"embeddings": embeddings,
                   "labels": labels,
                   "markers": markers}
    onpick_kwargs = {"subject_ids": subject_ids,
                     "embeddings": embeddings,
                     "report_df": report_df}

    # draw the initial scatterplot
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_latent_space(ax=ax, **plot_kwargs)

    # connect the click handler function to the scatterplot
    fig.canvas.mpl_connect('pick_event', lambda e: onpick(e, ax, **onpick_kwargs))

    # create the "clear all" button, and place it somewhere on the screen
    ax_clear_all = plt.axes([0.0, 0.0, 0.1, 0.05])
    button_clear_all = Button(ax_clear_all, 'Clear all')

    # link the event handler function to the click event on the button
    button_clear_all.on_clicked(lambda e: onclick(e, ax, plot_kwargs))

    # present the scatterplot
    plt.show()


def annotate(axis: plt.axes, text: str, x: float, y: float) -> None:
    """
    Adds a text annotation on the plot.
    This function is called within onpick to display subjects ids on the figure.
    """
    text_annotation = Annotation(text, xy=(x, y), xycoords='data')
    axis.add_artist(text_annotation)


def onpick(event: Any, ax: plt.axes, subject_ids: Any, embeddings: np.ndarray, report_df: pd.DataFrame) -> None:
    '''
    The behaviour taken when user picks a point on the scatterplot by clicking close to it.
    Required in interactive scatter plots.
    '''
    if subject_ids is None:
        raise RuntimeError("Subject Ids are not specified.")

    # actual coordinates of the click
    msx = event.mouseevent.xdata
    msy = event.mouseevent.ydata

    # select the closest index point
    ind = event.ind
    if len(ind) > 1:
        try:
            datax, datay = event.artist.get_data()
        except AttributeError:
            return
        datax, datay = [datax[i] for i in ind], [datay[i] for i in ind]
        dist = np.sqrt((np.array(datax) - msx) ** 2 + (np.array(datay) - msy) ** 2)
        ind = [ind[np.argmin(dist)]]
    ind = ind[0]

    # create and add the text annotation to the scatterplot
    annotate(ax, subject_ids[ind], msx, msy)
    if report_df is not None:
        print(report_df.query(f"USUBJID == '{subject_ids[ind]}'").drop(["ETHNIC", "TRTA", "TRTEYEN", "RGMCHG",
                                                                        "VISITNUM", "VISIT", "EXT1N", "OEDFN"], axis=1))

    # Plot indicator circle around the selected plot
    sc = event.artist.get_sizes()[0]
    ax.scatter(embeddings[ind, 0], embeddings[ind, 1], s=sc * 1.5, ec="black", color="none", lw=1)

    # force re-draw
    ax.figure.canvas.draw_idle()


def onclick(event: Any, ax: plt.axes, plot_args: Any) -> None:
    '''
    Repopulates the scatterplot on mouse click
    '''
    ax.cla()
    plot_latent_space(ax=ax, **plot_args)
    ax.figure.canvas.draw_idle()
