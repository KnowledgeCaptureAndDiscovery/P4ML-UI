from __future__ import print_function

import numpy as np
from scipy.linalg import orth
from sklearn.cluster import SpectralClustering
import stopit

from spider.clustering import ClusteringPrimitiveBase, TimeoutError
from ..kss import KSS
from typing import *

Inputs = np.ndarray
Outputs = np.ndarray
Params = NamedTuple('Params', [])


class EKSS(ClusteringPrimitiveBase[Inputs, Outputs, Params]):
    """ Ensemble K-Subspaces algorithm for subspace clustering

    """

    def __init__(self, n_clusters,
                 dim_subspaces=1,
                 n_base=100,
                 thresh=None):
        # type: (int, int, int, int) -> None
        """ Constructor for the EKSS class

        Arguments:
            n_clusters: The number of clusters
            dim_subspaces: dimension of subspaces(assumed all equal)
            n_base: number of base clusterings
            thresh: threshold parameter, integer, 0 < thresh <= n_samples
                    threshold the affinity matrix by taking top thresh values
                    from each row/column before applying Spectral Clustering
                    if thresh is None, no threshold applied
        Returns:
            None

        Raises:
            None
        """
        self._dim_subspaces = dim_subspaces
        self._n_base = n_base
        self._thresh = thresh
        self._max_iter = 1000
        self._k = n_clusters

    def produce(self, inputs, timeout=None, iterations=None):
        # type: (Inputs, float, int) -> Outputs
        """ Apply Ensemble K-Subspaces algorithm on D

        Arguments:
            inputs: array-like matrix, shape (n_samples, n_features) where
               n_samples is number of data points and
               n_features is ambient dimension
            timeout: This parameter serves as a way for caller to guide the length of the process
            iterations: This parameter serves as a way for caller to guide the length of the process

        Returns:
            estimated_labels: array of estimated labels, shape (n_samples)

        Raises:
            TimeoutError: If time taken for fitting the EKSS model exceeds the timeout variable
        """

        assert isinstance(inputs, (np.ndarray, np.generic, np.matrix)), "Input should be a numpy array"
        assert inputs.ndim == 2, "Data is not in the right shape"
        assert self._dim_subspaces <= inputs.shape[1], "Dim_subspaces should be less than ambient dimension"
        assert isinstance(self._k, int), "n_clusters is not integer"
        assert isinstance(self._dim_subspaces, int), "Dim_subspaces should be an integer"
        assert self._thresh <= inputs.shape[0], "Threshold should be in range 1:n_samples"

        _X = inputs.T
        n_features, n_samples = _X.shape

        self.affinity_matrix = np.zeros((n_samples, n_samples))

        with stopit.ThreadingTimeout(timeout) as to_ctx_mgr:
            # for each base clustering
            for b in range(self._n_base):
                # run K-Subspaces
                kss = KSS(n_clusters=self._k, dim_subspaces=self._dim_subspaces)
                est_labels = kss.produce(_X.T)
                # update affinity matrix
                for i in range(n_samples):
                    self.affinity_matrix[i][i] += 1
                    for j in range(i+1, n_samples):
                        if est_labels[i] == est_labels[j]:
                            self.affinity_matrix[i][j] += 1
                            self.affinity_matrix[j][i] += 1

            self.affinity_matrix = 1.0 * self.affinity_matrix / self._n_base

            # if thresh is not None, threshold affinity_matrix
            if self._thresh is not None:
                A_row = np.copy(self.affinity_matrix)
                A_col = np.copy(self.affinity_matrix.T)
                for i in range(n_samples):
                    # threshold rows
                    idx = np.argsort(A_row[i])[range(self._thresh)]
                    A_row[i][idx] = 0
                    # threshold columns
                    idx = np.argsort(A_col[i])[range(self._thresh)]
                    A_col[i][idx] = 0
                # average
                self.affinity_matrix = (A_row + A_col.T) / 2.0

            # apply Spectral Clustering with affinity_matrix
            sc = SpectralClustering(n_clusters= self._k, affinity='precomputed')
            estimated_labels = sc.fit_predict(self.affinity_matrix)

        if to_ctx_mgr.state == to_ctx_mgr.EXECUTED:
            return estimated_labels
        else:
            self.affinity_matrix = np.zeros((n_samples, n_samples))
            raise TimeoutError("EKSS fitting has timed out.")
