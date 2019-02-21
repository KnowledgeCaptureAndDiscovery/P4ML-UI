from __future__ import print_function

import numpy as np
from scipy.linalg import orth

from spider.clustering import ClusteringPrimitiveBase, TimeoutError
from typing import *
import stopit

Inputs = np.ndarray
Outputs = np.ndarray
Params = NamedTuple('Params', [])


class KSS(ClusteringPrimitiveBase[Inputs, Outputs, Params]):
    """ K-Subspaces algorithm for subspace clustering

    """

    def __init__(self, n_clusters,
                 dim_subspaces=1):
        # type: (int, int) -> None
        """ Constructor for the KSS class

        Arguments:
            n_clusters: The number of clusters
            dim_subspaces: dimension of subspaces (assumed all equal)

        Returns:
            None

        Raises:
            None
        """
        self._dim_subspaces = dim_subspaces
        self._max_iter = 1000
        self._k = n_clusters

    def produce(self, inputs, timeout=None, iterations=None):
        # type: (Inputs, float, int) -> Outputs
        """ Apply K-Subspaces algorithm on D

        Arguments:
            inputs: array-like matrix, shape (n_samples, n_features) where
               n_samples is number of data points and
               n_features is ambient dimension
            timeout: This parameter serves as a way for caller to guide the length of the process
            iterations: This parameter serves as a way for caller to guide the length of the process

        Returns:
            estimated_labels: array of estimated labels, shape (n_samples)

        Raises:
            TimeoutError: If time taken for fitting the KSS model exceeds the timeout parameter
        """

        assert isinstance(inputs, (np.ndarray, np.generic, np.matrix)), "Input should be a numpy array"
        assert inputs.ndim == 2, "Data is not in the right shape"
        assert self._dim_subspaces <= inputs.shape[1], "Dim_subspaces should be less than ambient dimension"
        assert isinstance(self._k, int), "n_clusters is not integer"
        assert isinstance(self._dim_subspaces, int), "Dim_subspaces should be an integer"

        _X = inputs.T
        n_features, n_samples = _X.shape

        with stopit.ThreadingTimeout(timeout) as to_ctx_mgr:
            # randomly initialize subspaces
            U_init = np.zeros((self._k, n_features, self._dim_subspaces))
            for kk in range(self._k):
                U_init[kk] = orth(np.random.randn(n_features, self._dim_subspaces))

            # compute residuals
            full_residuals = np.zeros((n_samples, self._k))
            for kk in range(self._k):
                tmp1 = np.dot(U_init[kk].T, _X)
                tmp2 = np.dot(U_init[kk], tmp1)
                full_residuals[:,kk] = np.linalg.norm(_X-tmp2, ord=2, axis=0)

            # label by nearest subspace
            estimated_labels = np.argmin(full_residuals, axis=1)

            # alternate between subspace estimation and assignment
            prev_labels = -1 * np.ones(estimated_labels.shape)
            it = 0
            while np.sum(estimated_labels != prev_labels) and it < self._max_iter:
                # first update residuals after labels obtained
                for kk in range(self._k):
                    Z = _X[:,estimated_labels == kk]
                    D, V = np.linalg.eig(np.dot(Z, Z.T))
                    D_idx = np.argsort(-D) # descending order
                    U = V[:,D_idx[range(self._dim_subspaces)]]
                    tmp1 = np.dot(U.T, _X)
                    tmp2 = np.dot(U, tmp1)
                    full_residuals[:,kk] = np.linalg.norm(_X-tmp2, ord=2, axis=0)
                # update prev_labels
                prev_labels = estimated_labels
                # label by nearest subspace
                estimated_labels = np.argmin(full_residuals, axis=1)

                it = it + 1

        if to_ctx_mgr.state == to_ctx_mgr.EXECUTED:
            return estimated_labels
        else:
            raise TimeoutError("KSS fitting has timed out.")
