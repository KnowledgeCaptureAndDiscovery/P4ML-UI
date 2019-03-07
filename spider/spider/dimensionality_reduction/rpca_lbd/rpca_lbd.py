##  @file rpca_lbd
#   implementation of RPCA_LBD class

import numpy as np
import numpy.matlib as ml
from numpy.linalg import norm
from numpy.linalg import svd
from sklearn.decomposition import PCA
from typing import *

from ...transformer import TransformerPrimitiveBase

Inputs = np.ndarray
Outputs = np.ndarray

##  RPCA_LBD class
#
#   uses RPCA-LBD to perform dimensionality reduction
class RPCA_LBD(TransformerPrimitiveBase[Inputs, Outputs]):
    
    __author__ = "Andrew Gitlin <agitlin@umich.edu>"
    __metadata__ = {
        "team": "Michigan",
        "common_name": "PCP IALM",
        "algorithm_type": ["Transformation"],
        "compute_resources": {
            "sample_size": [],
            "sample_unit": [],
            "disk_per_node": [],
            "expected_running_time": [],
            "gpus_per_node": [],
            "cores_per_node": [],
            "mem_per_gpu": [],
            "mem_per_node": [],
            "num_nodes": [],
        },
        "handles_regression": False,
        "handles_classification": False,
        "handles_multiclass": False,
        "handles_multilabel": False
    }

    ##  constructor for RPCA_LBD class
    #   @param name name associated with primitive
    #   @param kappa regularization parameter
    #   @param lamb regularization parameter
    #   @param rho outer loop parameter
    #   @param beta inner loop parameter
    #   @param alpha inner loop parameter
    #   @param outer_epsilon termination constant for outer loop
    #   @param outer_max_iter maximum number of iterations for outer loop
    #   @param inner_epsilon termination constant for inner loop
    #   @param inner_max_iter maximum number of iterations for inner loop
    def __init__(self, kappa = 1.1, lamb = 0.61, rho = 1.1, beta = 0.2, alpha = 1.0,
             outer_epsilon = 1e-7, outer_max_iter = 500, inner_epsilon = 1e-6, inner_max_iter = 20):
        # type: (float, float, float, float, float, float, int, float, int) -> None
        self._kappa = kappa
        self._lamb = lamb
        self._rho = rho
        self._beta = beta
        self._alpha = alpha
        self._epsilon = [outer_epsilon, inner_epsilon]
        self._max_iter = [outer_max_iter, inner_max_iter]

    ##  RPCA based on the Low-Rank and Block-Sparse Decomposition (LBD) model
    #   @param data data matrix (NumPy array/matrix where rows are samples and columns are features)
    #   @return W low-rank component of data matrix (NumPy array with same shape as data matrix)
    def rpca(self, data):
        def SVD_thresh(X, tau):
            U,s,V = svd(X, full_matrices = False)
            s_thresh = np.array([max(abs(sig) - tau, 0.0) * np.sign(sig) for sig in s])
            return U * np.matrix(np.diag(s_thresh)) * V
        
        def col_thresh(X, tau):
            C = ml.zeros(X.shape)
            for i in range(X.shape[1]):
                Xi_norm = norm(X[:,i],2)
                if Xi_norm > tau:
                    C[:,i] = X[:,i] * (1.0 - (tau / Xi_norm))
            return C
        
        D = np.transpose(np.matrix(data))
        D_norm = norm(D,'fro')
        A = np.matrix(D)
        E = ml.zeros(D.shape)
        Y = ml.zeros(D.shape)
        mu = 30.0 / norm(np.sign(D),2)
        kappa = self._kappa
        lamb = self._lamb
        rho = self._rho
        beta = self._beta
        alpha = self._alpha
        err_outer = 10.0 * self._epsilon[0] * D_norm
        k = 0
        num_iter = 0
        while k < self._max_iter[0] and err_outer > self._epsilon[0] * D_norm:
            GA = D - E + Y / mu
            A = GA
            A_save = ml.zeros(D.shape)
            err_inner = 10.0 * self._epsilon[1]
            j = 0
            while j < self._max_iter[1] and err_inner > self._epsilon[1]:
                A_save = SVD_thresh(A,beta)
                A_old = A
                Mat = (2.0 * A_save - A + beta * mu * GA) / (1.0 + beta * mu)
                A = A + alpha * (col_thresh(Mat, beta*kappa*(1.0-lamb)/(1.0+beta*mu)) - A_save)
                err_inner = norm(A-A_old,'fro')
                j += 1
            num_iter += j
            A = A_save
            GE = D - A + Y / mu
            E = col_thresh(GE, kappa*lamb/mu)
            Y = Y + mu * (D-A-E)
            mu *= rho
            err_outer = norm(D-A-E,'fro')
            k += 1
        W = np.array(A.T)
        return W

    ##  dimensionality reduction with RPCA-LBD
    #   @param A collection of vectors in high dimensional space where rows are samples and columns are features (duck-typed)
    #   @return W collection of vectors in low dimensional space where rows are samples and columns are features (NumPy array)
    def produce(self, A, timeout = None, iterations = None):
        # type: (Inputs, float, int) -> Outputs
        return self.rpca(A)

