##  @file go_dec
#   implementation of GO_DEC class

import numpy as np
import numpy.matlib as ml
from numpy.linalg import norm
from scipy.linalg import qr
from sklearn.decomposition import PCA
from typing import *

from ...transformer import TransformerPrimitiveBase

Inputs = np.ndarray
Outputs = np.ndarray

##  GO_DEC class
#
#   uses RPCA via the Go Decomposition (GoDec) to perform dimensionality reduction
class GO_DEC(TransformerPrimitiveBase[Inputs, Outputs]):
    
    __author__ = "Andrew Gitlin <agitlin@umich.edu>"
    __metadata__ = {
        "team": "Michigan",
        "common_name": "GoDec",
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

    ##  constructor for GO_DEC class
    #   @param name name associated with primitive
    #   @param c maximum cardinality of sparse component (if c>=1 then max card = c; if 0<c<1 then max card = c*m*n where (m,n) is shape of data matrix)
    #   @param r maximum rank of low-rank component (if r>=1 then max rank = r; if 0<r<1 then max rank = r*min(m,n) where (m,n) is shape of data matrix)
    #   @param power power scheme modification (larger power leads to higher accuracy and higher time cost)
    #   @param epsilon termination constant
    #   @param max_iter maximum number of iterations
    def __init__(self, c = 0.03, r = 0.1,
                 power = 2, epsilon = 1e-3, max_iter = 100):
        # type: (float, float, int, float, int) -> None
        self._card = c
        self._r = r
        self._power = power
        self._epsilon = epsilon
        self._max_iter = max_iter
    
    ##  RPCA via the Go Decomposition (GoDec)
    #   @param data data matrix (NumPy array/matrix where rows are samples and columns are features)
    #   @return W low-rank component of data matrix (NumPy array with same shape as data matrix)
    def rpca(self, data):
        
        def largest_entries(M, kk):
            k = int(kk)
            L = np.array(np.absolute(M)).ravel()
            ind = np.argpartition(L, -k)[-k:]
            Y = np.zeros(len(L))
            LL = np.array(M).ravel()
            Y[ind] = LL[ind]
            return np.matrix(Y.reshape(M.shape))
        
        X = np.matrix(np.transpose(data))
        m,n = X.shape
        rank = self._r if self._r >= 1 else np.floor(self._r * min(m, n))
        card = self._card if self._card >= 1 else np.floor(self._card * m * n)
        r = self._r
        L = np.matrix(X)
        S = ml.zeros(X.shape)
        itr = 1
        while True:
            Y2 = np.matrix(np.random.normal(size = (n,int(rank))))
            for i in range(self._power + 1):
                Y1 = L * Y2
                Y2 = L.T * Y1
            Q,R = qr(Y2,mode='economic')
            L_new = L * Q * Q.T
            T = L - L_new + S
            L = L_new
            S = largest_entries(T,card)
            T = T - S
            if norm(T,'fro') < self._epsilon or itr > self._max_iter:
                break
            L = L + T
            itr += 1
        W = np.array(L.T)
        return W

    ##  dimensionality reduction with RPCA via GoDec
    #   @param A collection of vectors in high dimensional space where rows are samples and columns are features (duck-typed)
    #   @return W collection of vectors in low dimensional space where rows are samples and columns are features (NumPy array)
    def produce(self, A, timeout = None, iterations = None):
        # type: (Inputs, float, int) -> Outputs
        return self.rpca(A)

