'''
    ssc_omp.py
    @author akash
    Implements the SSC using Orthogonal Matching Pursuit Algorithm
'''

import numpy as np
import scipy as sp
from scipy.sparse.linalg import svds, eigs
from scipy.linalg import lstsq
import scipy.sparse as sps
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

import stopit
from spider.clustering import ClusteringPrimitiveBase, TimeoutError
from typing import *

Inputs = np.ndarray
Outputs = np.ndarray
Params = NamedTuple('Params', [])


class SSC_OMP(ClusteringPrimitiveBase[Inputs, Outputs, Params]):
    _epsL = np.finfo(np.float32).eps

    def __init__(self, n_clusters, max_subspace_dim=6):
        # type: (int, int) -> None
        """
        Constructor ssc_omp class
        :param n_clusters: The number of clusters
        :param max_subspace_dim: the maximum dimenstion of the subspace
        """
        self.max_subspace_dim = max_subspace_dim
        self.thres = 1e-6
        self.max_iter = 1000
        self.n_init = 20
        self._k = n_clusters


    @staticmethod
    def _cNormalize(data, norm=2):
        """
        This method performs the column wise normalization of the input data
        :param data: A dxN numpy array
        :param norm: the desired norm value (This has to be in accordance with the accepted numpy
         norm values
        :return: Returns the column wise normalised data
        """
        return data / (np.linalg.norm(data, ord=norm, axis = 0) + SSC_OMP._epsL)

    @staticmethod
    def _OMPMatFunction(data, K, thres):
        """
        This code implements the subspace clustering algorithm described in
        Chong You, Daniel Robinson, Rene Vidal,
        "Scalable Sparse Subspace Clustering by Orthogonal Matching Pursuit", CVPR 2016.

        It performs the OMP algorithm on every column of X using all other columns as a
        dicitonary

        :param data: A dxN numpy array
        :param K: The maximum subspace dimension
        :param thres: termination condition
        :return: the SSC-OMP representation of the data
        """
        memory_total = 0.1 * 10**9
        _, n = data.shape
        data_normalised = SSC_OMP._cNormalize(data)
        support_set = np.ones((n, K), dtype=np.int64)
        indices = np.arange(n, dtype=np.int64).reshape(n, 1) * np.ones((1, K))
        values = np.zeros((n, K))
        t_vector = np.ones((n, 1), dtype=np.int64) * K
        residual = np.copy(data_normalised)

        for t in range(K):
            counter = 0
            block_size = np.ceil(memory_total / n)
            while True:
                mask = np.arange(counter, min(counter+block_size, n))
                iMat = np.abs(np.matmul(data.T, residual[:, mask]))
                np.fill_diagonal(iMat, 0.0)
                jMat = np.argmax(iMat, axis=0)
                support_set[mask, t] = jMat
                counter = counter + block_size
                if counter >= n:
                    break

            if t+1 != K:
                for iN in range(n):
                    if t_vector[iN] == K:
                        B = data_normalised[:, support_set[iN, 0:(t+1)]]
                        mat_tmp, _, _, _ = lstsq(B, data_normalised[:, iN])

                        residual[:, iN] = data_normalised[:, iN] - np.matmul(B, mat_tmp)

                        if np.sum(residual[:, iN]**2) < thres:
                            t_vector[iN] = t

            if not np.any(K == t_vector):
                break

        for iN in range(n):
            tmp, _, _, _ = lstsq(data[:, support_set[iN, 0:np.asscalar(t_vector[iN] + 1)]], (data[:, iN]))
            values[iN, 0:np.asscalar(t_vector[iN])] = tmp.T

        sparse_mat = sps.coo_matrix((values.flat, (support_set.flat, indices.flat)), shape=(n, n))
        sparse_mat = sparse_mat.toarray()
        return sparse_mat

    @staticmethod
    def _spectral_clustering(W, n_clusters=10, max_iter=1000, n_init=20):
        """
        This method performs the spectral clustering on the affinity matrix W.
        :param W: Affinity matrix
        :param n_clusters: Desired number of clusters
        :param max_iter: The maximum number of iterations of kMeans algorithm
        :param n_init: The number of times kMeans algorithm should run
        :return: Returns the cluster assignments as per the spectral clustering algorithm
        """
        n, _ = W.shape
        eps = 2.220446049250313e-16
        dn = np.diag(1 / np.sqrt(np.sum(W, axis=0) + eps))
        lapN = np.identity(n) - np.matmul(np.matmul(dn, W), dn)
        _, _, vn = np.linalg.svd(lapN)
        kerN = vn.T[:, (n - n_clusters):n]
        normN = np.sqrt(np.sum(np.square(kerN), axis=1))
        kerNS = (kerN.T / (normN + eps).T).T
        l = KMeans(n_clusters, n_init=n_init, max_iter = max_iter).fit(kerNS)
        labels = l.labels_.reshape((n,))
        return labels

    def produce(self, inputs, timeout=None, iterations=None):
        # type: (Inputs, float, int) -> Ouputs
        """
        This method performs the following steps:
          1) Gets the SSC-OMP representation
          2) Uses the SSC-OMP representation to generate Affinity matrix
          3) Use spectral clustering to assign labels
        :param inputs: A NxD numpy array of input features
        :param timeout: This parameter serves as a way for caller to guide the length of the process
        :param iterations: This parameter serves as a way for caller to guide the length of the process
        :return: labels: A numpy array with label assignment for each data point
        :raise: TimeoutError: if the time taken for ssc_omp fitting is more than the timeout parameter
        """
        assert isinstance(inputs, (np.ndarray, np.matrix, np.generic)), "Data is not numpy array"
        assert inputs.ndim == 2, "Data is not in the right shape"
        assert isinstance(self._k, int), "n_clusters is not integer"
        assert isinstance(self.max_subspace_dim, int), "max_subspace_dim should be integer"
        assert isinstance(self.max_iter, int), "max_iter should be integer"
        assert isinstance(self.n_init, int), "n_init should be integer"
        assert self.max_subspace_dim <= inputs.shape[1], "max_subspace dim can't be greater than the" + \
        "input feature space"

        with stopit.ThreadingTimeout(timeout) as to_ctx_mgr:
            data = inputs.T
            R = SSC_OMP._OMPMatFunction(data, self.max_subspace_dim, self.thres)
            np.fill_diagonal(R, 0)
            A = np.abs(R) + np.abs(R.T)
            labels = SSC_OMP._spectral_clustering(A, n_clusters=self._k, max_iter=self.max_iter, n_init=self.n_init)

        if to_ctx_mgr.state == to_ctx_mgr.EXECUTED:
            return labels
        else:
            raise TimeoutError("SSC OMP fitting has timed out.")
