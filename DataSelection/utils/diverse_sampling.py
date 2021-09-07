#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

from dataclasses import dataclass
from enum import Enum
from typing import List, Set

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances


class DistanceMetric(Enum):
    Cosine = "cosine"
    Euclidean = "Euclidean"


@dataclass(init=True, frozen=True)
class DiverseSamplingParameters:
    """
    Class for setting graph connectivity and diffusion parameters.
    """
    n_clusters: int
    diverse_oversampling_factor: float
    seed: int = 42
    distance_metric: DistanceMetric = DistanceMetric.Cosine


@dataclass
class Cluster:
    """
    A data class to represent clusters of embedding points.
    """
    index: int
    elements: List[int]
    visited: bool

    def get_size(self) -> int:
        return len(self.elements)

    def is_empty(self) -> bool:
        return self.get_size() == 0


def diverse_sample(case_ind: np.ndarray,
                   embeddings: np.ndarray,
                   batch_size: int,
                   diverse_sampling_parameters: DiverseSamplingParameters) -> np.ndarray:
    """
    Function to perform diverse sampling of extracted easy, difficult, and label noise cases.
    The embeddings are clustered using the k-means algorithm and indices closest to cluster centres are returned.

    :param case_ind:   A list of indices selected by the uncertainty quantification algorithm
    :param embeddings: Lower dimensional embeddings of all data points, which will be clustered.
    :param batch_size: The required maximum batch_size. If there are not enough points, the function returns
                       input indices since there is no need for diverse sampling.
    :param diverse_sampling_parameters: Parameter defining the diverse sampling (n_cluster, seed, distance_metric)
    """
    if diverse_sampling_parameters.distance_metric == DistanceMetric.Cosine:
        metric = cosine_distances
    elif diverse_sampling_parameters.distance_metric == DistanceMetric.Euclidean:
        metric = euclidean_distances
    else:
        raise ValueError("The provided distance metric is invalid.")

    def find_closest(cluster: np.ndarray, center: np.ndarray) -> np.ndarray:
        return np.argmin(metric(embeddings[cluster, :], Y=center))

    # If the number of candidates is smaller than the requested size, return the input indices.
    if len(case_ind) < batch_size:
        return case_ind

    # Run the clustering algorithm
    _kmeans = KMeans(n_clusters=diverse_sampling_parameters.n_clusters,
                     init='k-means++',
                     max_iter=300,
                     random_state=diverse_sampling_parameters.seed)
    kmeans_labels = _kmeans.fit_predict(embeddings)
    np.random.shuffle(case_ind)
    clusters = dict()
    filtered_case_ind: Set[int] = set()

    # Extract clustered indices
    for _l in np.sort(np.unique(kmeans_labels)):
        inter = np.intersect1d(case_ind, np.where(kmeans_labels == _l)[0])
        clusters[_l] = Cluster(index=_l, elements=inter.tolist(), visited=False)

    # Filter out indices from clusters
    _cur = 0
    while len(filtered_case_ind) < batch_size:
        if not clusters[_cur].is_empty():
            selected = find_closest(clusters[_cur].elements,
                                    _kmeans.cluster_centers_[[_cur], :]) if not clusters[_cur].visited else -1
            filtered_case_ind.add(clusters[_cur].elements.pop(selected))
            clusters[_cur].visited = True
        _cur = (_cur + 1) % len(clusters)

    return np.asarray(list(filtered_case_ind))
