##  @file pcp_ialm
#   implementation of PCP_IALM class

import numpy as np
import numpy.matlib as ml
from numpy.linalg import norm
from numpy.linalg import svd
from sklearn.decomposition import PCA
from typing import *

from ...transformer import TransformerPrimitiveBase

Inputs = np.ndarray
Outputs = np.ndarray

##  PCP_IALM class
#
#   uses RPCA via PCP-IALM to perform dimensionality reduction
class PCP_IALM(TransformerPrimitiveBase[Inputs, Outputs]):
    
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

    ##  constructor for PCP_IALM class
    #   @param name name associated with primitive
    #   @param lamb regularization parameter for sparse component
    #   @param mu penalty parameter in Lagrangian function for noise
    #   @param rho constant used to update mu in each iteration
    #   @param epsilon termination constant
    #   @param max_iter maximum number of iterations
    def __init__(self, lamb = -1, mu = -1,
                 rho = 1.5, epsilon = 1e-7, max_iter = 1000):
        # type: (float, float, float, float, int) -> None
        self._lamb = lamb
        self._mu = mu
        self._rho = rho
        self._epsilon = epsilon
        self._max_iter = max_iter
    
    ##  RPCA via Principal Component Pursuit (PCP) with the Inexact Augmented Lagrange Multiplier (IALM) method
    #   @param data data matrix (NumPy array/matrix where rows are samples and columns are features)
    #   @return W low-rank component of data matrix (NumPy array with same shape as data matrix)
    def rpca(self, data):
        
        def shrink(T, zeta):
            return np.matrix(np.maximum(np.zeros(T.shape), np.array(np.absolute(T)) - zeta) * np.array(np.sign(T)))
        
        def SVD_thresh(X, tau):
            U,s,V = svd(X, full_matrices = False)
            s_thresh = np.array([max(abs(sig) - tau, 0.0) * np.sign(sig) for sig in s])
            return U * np.matrix(np.diag(s_thresh)) * V
        
        D = np.transpose(np.matrix(data))
        d,n = data.shape
        k = 0
        D_norm_fro = norm(D,'fro')
        D_norm_2 = norm(D,2)
        D_norm_inf = norm(D,np.inf)
        mu = self._mu if self._mu != -1 else 1.25 / D_norm_2
        rho = self._rho
        lamb = self._lamb if self._lamb != -1 else 1.0 / np.sqrt(d)
        Y = D / max(D_norm_2, D_norm_inf / lamb)
        E = ml.zeros(D.shape)
        A = ml.zeros(D.shape)
        while k < self._max_iter and norm(D-A-E,'fro') > self._epsilon * D_norm_fro:
            A = SVD_thresh(D - E + Y / mu, 1 / mu)
            E = shrink(D - A + Y / mu, lamb / mu)
            Y = Y + mu * (D - A - E)
            mu = rho * mu
            k += 1
        W = np.array(A.T)
        return W

    ##  dimensionality reduction with RPCA via PCP-IALM
    #   @param A collection of vectors in high dimensional space where rows are samples and columns are features (duck-typed)
    #   @return W collection of vectors in low dimensional space where rows are samples and columns are features (NumPy array)
    def produce(self, A, timeout = None, iterations = None):
        # type: (Inputs, float, int) -> Outputs
        return self.rpca(A)

