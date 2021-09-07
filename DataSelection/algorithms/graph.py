#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import scipy
from scipy.special import softmax
from sklearn.neighbors import kneighbors_graph


@dataclass(init=True, frozen=True)
class GraphParameters:
    """Class for setting graph connectivity and diffusion parameters."""
    n_neighbors: int
    diffusion_alpha: float
    cg_solver_max_iter: int
    diffusion_batch_size: Optional[int]
    distance_kernel: str  # {'euclidean' or 'cosine'}


def _get_affinity_matrix(embeddings: np.ndarray,
                         n_neighbors: int,
                         distance_kernel: str = 'cosine') -> scipy.sparse.csr.csr_matrix:
    """
    :param embeddings: Input sample embeddings (n_samples x n_embedding_dim)
    :param n_neighbors: Number of neighbors in the KNN Graph
    :param distance_kernel: Distance kernel to compute sample similarities {'euclidean' or 'cosine'}
    """

    # Build a k-NN graph using the embeddings
    if distance_kernel == 'euclidean':
        sigma = embeddings.shape[1]
        knn_dist_graph = kneighbors_graph(embeddings, n_neighbors, mode='distance', metric='euclidean', n_jobs=-1)
        knn_dist_graph.data = np.exp(-1.0 * np.asarray(knn_dist_graph.data)**2 / (2.0 * sigma**2))
    elif distance_kernel == 'cosine':
        knn_dist_graph = kneighbors_graph(embeddings, n_neighbors, mode='distance', metric='cosine', n_jobs=-1)
        knn_dist_graph.data = 1.0 - np.asarray(knn_dist_graph.data)
    else:
        raise ValueError(f"Unknown sample distance kernel {distance_kernel}")

    return knn_dist_graph


def build_connectivity_graph(normalised: bool = True, **affinity_kwargs: Any) -> np.ndarray:
    """
    Builds connectivity graph and returns adjacency and degree matrix
    :param normalised: If set to True, graph adjacency is normalised with the norm of degree matrix
    :param affinity_kwargs: Arguments required to construct an affinity matrix
                            (weights representing similarity between points)
    """

    # Build a symmetric adjacency matrix
    A = _get_affinity_matrix(**affinity_kwargs)
    W = 0.5 * (A + A.T)
    if normalised:
        # Normalize the similarity graph
        W = W - scipy.sparse.diags(W.diagonal())
        D = W.sum(axis=1)
        D[D == 0] = 1
        D_sqrt_inv = np.array(1. / np.sqrt(D))
        D_sqrt_inv = scipy.sparse.diags(D_sqrt_inv.reshape(-1))
        L_norm = D_sqrt_inv * W * D_sqrt_inv
        return L_norm
    else:
        num_samples = W.shape[0]
        D = W.sum(axis=1)
        D = np.diag(np.asarray(D).reshape(num_samples,))
        L = D - W
        return L


def label_diffusion(adjacency: np.ndarray,
                    labels: np.ndarray,
                    query_batch_ids: np.ndarray,
                    graph_params: GraphParameters,
                    class_priors: Optional[np.ndarray] = None,
                    diffusion_normalizing_factor: float = 0.01) -> np.ndarray:
    """
    :param adjacency: adjacency matrix of the graph
    :param labels:
    :param query_batch_ids: the ids of the "labeled" samples
    :param graph_params: parameter to construct the graph
    :param class_priors: prior distribution of each class [n_classes,]
    :param diffusion_normalizing_factor: factor to normalize the diffused labels
    """
    diffusion_start = time.time()

    # Input number of nodes and classes
    n_samples = labels.shape[0]
    n_classes = labels.shape[1]

    # Find the labelled set of nodes in the graph
    all_idx = np.array(range(n_samples))
    labeled_idx = np.setdiff1d(all_idx, query_batch_ids.flatten())
    assert(np.all(np.isin(query_batch_ids, all_idx)))
    assert(np.allclose(np.sum(labels, axis=1), np.ones(shape=(labels.shape[0])), rtol=1e-3))

    # Initialize the y vector for each class (eq 5 from the paper, normalized with the class size)
    # and apply label propagation
    Z = np.zeros((n_samples, n_classes))
    L = scipy.sparse.eye(adjacency.shape[0]) - graph_params.diffusion_alpha * adjacency

    for class_id in range(n_classes):
        # Identify indices with non-zero labels and assign their probability value to vector y
        cur_idx = labeled_idx[np.where(labels[labeled_idx, class_id] != 0.0)[0]]
        y = np.zeros((n_samples,))
        y[cur_idx] = labels[cur_idx, class_id] / np.sum(labels[cur_idx, class_id])
        if class_priors is not None:
            y[cur_idx] = y[cur_idx] * class_priors[class_id]
        f, _ = scipy.sparse.linalg.cg(L, y, tol=1e-6, maxiter=graph_params.cg_solver_max_iter)
        Z[:, class_id] = f

    # Normalise the diffused logits
    # TODO: How do we find the right normalising factor in here
    output = softmax(Z[query_batch_ids, :] / diffusion_normalizing_factor, axis=1)

    logging.debug(f"Graph diffusion time: {0: .2f} secs".format(time.time() - diffusion_start))

    return output