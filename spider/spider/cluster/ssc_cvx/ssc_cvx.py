##  @file ssc_cvx
#   implementation of SSC_CVX class

import stopit
import numpy as np
import numpy.matlib
from sklearn.cluster import KMeans
from cvxpy import *

from spider.clustering import ClusteringPrimitiveBase, TimeoutError
from typing import *

Inputs = np.ndarray
Outputs = np.ndarray
Params = NamedTuple('Params', [])

##  SSC_CVX class
#
#   uses SSC algorithm with convex optimization to perform subspace clustering
#   compute_sparse_coefficient_matrix computes sparse coefficient matrix
#   fit_predict computes predicted labels
#   cluster returns predicted clustering
class SSC_CVX(ClusteringPrimitiveBase[Inputs, Outputs, Params]):
    
    ##  constructor for SSC_CVX class
    #   @param use_affine whether or not data points come from union of affine subspaces instead of linear subspaces
    #   @param use_outliers whehter or not to use outliers
    #   @param use_noise whether or not to use noise
    #   @param alpha constant used in calculating updates
    def __init__(self, n_clusters, use_affine = False, use_outliers = True, use_noise = True, alpha = -1):
        # type: (int, bool, bool, bool, float) -> None
        self._use_affine = use_affine
        self._use_outliers = use_outliers
        self._use_noise = use_noise
        self._alpha = alpha
        self._k = n_clusters

    ##  computes adjacency matrix given coefficient matrix
    #   @param C NxN coefficient matrix (NumPy matrix)
    #   @return NxN adjacency matrix (NumPy matrix)
    def build_adjacency_matrix(self, C):
        eps = 2.220446049250313e-16
        N = C.shape[0]
        CAbs = np.absolute(C)
        for i in range(N):
            CAbs[:,i] = CAbs[:,i] / (np.amax(CAbs[:,i]) + eps)
        A = CAbs + np.transpose(CAbs) + eps
        np.fill_diagonal(A,0.0)
        return A

    ##  spectral clustering algorithm
    #   @param W NxN adjacency matrix (NumPy matrix)
    #   @param n_clusters number of clusters
    #   @param max_iter maximum number of iterations for KMeans
    #   @param n_init number of replications for KMeans
    #   @return labels for N points
    def spectral_clustering(self, W, n_clusters = 10, max_iter = 1000, n_init = 20):
        N,_ = W.shape
        eps = 2.220446049250313e-16
        DN = np.diag(1/np.sqrt(np.sum(W, axis = 0) + eps))
        LapN = np.identity(N) - np.matmul(np.matmul(DN, W), DN)
        _, _, VN = np.linalg.svd(LapN)
        kerN = VN.T[:,(N - n_clusters):N]
        normN = np.sqrt(np.sum(np.square(kerN), axis = 1));
        kerNS = (kerN.T / (normN + eps).T).T
        l = KMeans(n_clusters, n_init = n_init, max_iter = max_iter).fit(kerNS)
        labels = l.labels_.reshape((N,))
        return labels

    ##  computes sparse coefficient matrix using SSC algorithm with convex optimization
    #   @param X NxD NumPy array/matrix representing N points in D-dimensional space
    #   @return sparse coefficient matrix
    def compute_sparse_coefficient_matrix(self, X):
    
        Y = np.transpose(np.array(X))
        D,N = Y.shape
        a = self._alpha if self._alpha != -1 else (20.0 if self._use_outliers else 800.0)
        le = a / np.min([np.max([np.linalg.norm(Y[:,j],1) for j in range(N) if j != i]) for i in range(N)]) if self._use_outliers else 0.0
        lz = a / np.min([np.max([np.absolute(np.dot(Y[:,i],Y[:,j])) for j in range(N) if j != i]) for i in range(N)]) if self._use_noise else 0.0
        C = np.zeros((N,N))
    
        # find sparse coefficient matrix C using convex optimization
        for i in range(N):
            # since cii = 0, we can treat ci as (N-1)x1 vector
            ci = Variable(N-1,1)
            ei = Variable(D,1)
            zi = Variable(D,1)
            Yi = np.delete(Y,i,axis=1)
            yi = Y[:,i]
            objective = None
            constraints = []
            if self._use_outliers and self._use_noise:
                objective = Minimize(norm(ci,1) + le * norm(ei,1) + 0.5 * lz * (norm(zi,2) ** 2))
                constraints = [yi == Yi * ci + ei + zi]
            elif self._use_outliers and not self._use_noise:
                objective = Minimize(norm(ci,1) + le * norm(ei,1))
                constraints = [yi == Yi * ci + ei]
            elif not self._use_outliers and self._use_noise:
                objective = Minimize(norm(ci,1) + 0.5 * lz * (norm(zi,2) ** 2))
                constraints = [yi == Yi * ci + zi]
            else:
                objective = Minimize(norm(ci,1))
                constraints = [yi == Yi * ci]
            if self._use_affine:
                constraints.append(np.ones((1,N-1)) * ci == 1)
            prob = Problem(objective, constraints)
            result = prob.solve()
            # turn ci into Nx1 vector by setting cii = 0 and set column i of C equal to ci
            C[:,i] = np.insert(np.asarray(ci.value),i,0,axis=0)[:,0]

        return C
   
    ##  computes predicted labels using SSC algorithm with convex optimization
    #   @param inputs nxd NumPy array/matrix representing n points in d-dimensional space
    #   @param timeout This parameter serves as a way for caller to guide the length of the process
    #   @param iterations This parameter serves as a way for caller to guide the length of the process
    #   @return predicted labels
    def produce(self, inputs, timeout=None, iterations=None):
        # type: (Inputs, float, int) -> Outputs
        with stopit.ThreadingTimeout(timeout) as to_ctx_mgr:
            C = self.compute_sparse_coefficient_matrix(inputs)
            W = self.build_adjacency_matrix(C)
            labels = self.spectral_clustering(W, n_clusters = self._k)
            labels = np.array(labels)

        if to_ctx_mgr.state == to_ctx_mgr.EXECUTED:
            return labels
        else:
            raise TimeoutError("SSC CVX fitting has timed out.")

