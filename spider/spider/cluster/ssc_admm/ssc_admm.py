##  @file ssc_admm
#   implementation of SSC_ADMM class

import stopit
import numpy as np
import numpy.matlib
from sklearn.cluster import KMeans

from spider.clustering import ClusteringPrimitiveBase, TimeoutError
from typing import *


Inputs = np.ndarray
Outputs = np.ndarray
Params = NamedTuple('Params', [])

##  SSC_ADMM class
#
#   uses SSC algorithm with ADMM to perform subspace clustering
#   compute_sparse_coefficient_matrix computes sparse coefficient matrix
#   fit_predict computes predicted labels
#   cluster returns predicted clustering


class SSC_ADMM(ClusteringPrimitiveBase[Inputs, Outputs, Params]):
    
    ##  constructor for SSC_ADMM class
    #   @param n_clusters = number of clusters
    #   @param use_affine whether or not data points come from union of affine subspaces instead of linear subspaces
    #   @param use_outliers whehter or not to use outliers
    #   @param alpha constant used in calculating updates
    def __init__(self, n_clusters, use_affine=False, use_outliers=True, alpha=-1):
        # type: (int, bool, bool, bool, float) -> None
        self._use_affine = use_affine
        self._use_outliers = use_outliers
        self._alpha = alpha if alpha != -1 else (20 if self._use_outliers else 800)
        self._epsilon = 0.0002
        self._max_iter = 150 if self._use_outliers else 200
        self._k = n_clusters
    
    ##  computes regularization paramater lambda to be used in ADMM algorithm
    #   @param Y DxN data matrix
    #   @param P Dx? modified data matrix
    #   @return regularization paramater lambda for ADMM algorithm
    def compute_lambda(self, Y, P):
        T = P.T * Y
        np.fill_diagonal(T,0.0)
        T = np.absolute(T)
        l = np.min(np.amax(T, axis = 0))
        return l

    ##  shrinkage threshold operator
    #   @param eta number
    #   @param M NumPy matrix
    #   @return NumPy matrix resulting from applying shrinkage threshold operator to each entry of M
    def shrinkage_threshold(self, eta, M):
        ST = np.matrix(np.maximum(np.zeros(M.shape), np.array(np.absolute(M)) - eta) * np.array(np.sign(M)))
        return ST

    ##  computes maximum L2-norm error among columns of residual of linear system
    #   @param P DxN NumPy matrix
    #   @param Z NxN NumPy matrix
    #   @return maximum L2-norm of columns of P-P*Z
    def error_linear_system(self, P, Z):
        R,N = Z.shape
        Y = P[:,:N] if R > N else P
        Y0 = Y - P[:,N:] * Z[N:,:] if R > N else P
        C = Z[:N,:] if R > N else Z
        n = np.linalg.norm(Y0, 2, axis = 0)
        S = np.array((Y0 / n) - Y * (C / n))
        err = np.sqrt(np.max(sum(S*S)))
        return err

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
        normN = np.sqrt(np.sum(np.square(kerN), axis = 1))
        kerNS = (kerN.T / (normN + eps).T).T
        l = KMeans(n_clusters, n_init = n_init, max_iter = max_iter).fit(kerNS)
        labels = l.labels_.reshape((N,))
        return labels

    ##  ADMM algorithm with outliers
    #   @param X DxN NumPy array/matrix representing N points in D-dimensional space
    #   @param use_affine whether or not data points come from union of affine subspaces instead of linear subspaces
    #   @param alpha constant used in calculating updates
    #   @param epsilon termination constant
    #   @param max_iter maximum number of iterations
    #   @return sparse coefficient matrix (NumPy array)
    def outlier_admm(self, X, use_affine = False, alpha = 20.0, epsilon = 0.0002, max_iter = 150):
    
        Y = np.matrix(X)
        D,N = Y.shape
        gamma = alpha / np.linalg.norm(Y,1)
        P = np.concatenate((Y, np.matlib.eye(D) / gamma), axis = 1)
        mu1 = alpha / self.compute_lambda(Y,P)
        mu2 = alpha
        C = np.matlib.zeros((N+D,N))
    
        if not use_affine:
        
            # initializations
            k = 1
            A = np.linalg.pinv(mu1*P.T*P + mu2*np.matlib.eye(N+D))
            Lambda1 = np.matlib.zeros((D,N))
            Lambda2 = np.matlib.zeros((N+D,N))
            err1 = 10.0 * epsilon
            err2 = 10.0 * epsilon
        
            # main loop
            while k < max_iter and (err1 > epsilon or err2 > epsilon):
                Z = A * (mu1*P.T*(Y+Lambda1/mu1) + mu2*(C-Lambda2/mu2))
                np.fill_diagonal(Z,0.0)
                C = self.shrinkage_threshold(1.0/mu2, Z+Lambda2/mu2)
                np.fill_diagonal(C,0.0)
                Lambda1 = Lambda1 + mu1 * (Y - P * Z)
                Lambda2 = Lambda2 + mu2 * (Z - C)
                err1 = np.amax(np.absolute(Z-C))
                err2 = self.error_linear_system(P,Z)
                k += 1

        else:
    
            # initializations
            k = 1
            delta = np.matrix([[float(i < N)] for i in range(N+D)])
            A = np.linalg.pinv(mu1*P.T*P + mu2*np.matlib.eye(N+D) + mu2*delta*delta.T)
            Lambda1 = np.matlib.zeros((D,N))
            Lambda2 = np.matlib.zeros((N+D,N))
            lambda3 = np.matlib.zeros((1,N))
            err1 = 10.0 * epsilon
            err2 = 10.0 * epsilon
            err3 = 10.0 * epsilon
        
            # main loop
            while k < max_iter and (err1 > epsilon or err2 > epsilon or err3 > epsilon):
                Z = A * (mu1*P.T*(Y+Lambda1/mu1) + mu2*(C-Lambda2/mu2) + mu2*delta*(1.0-lambda3/mu2))
                np.fill_diagonal(Z,0.0)
                C = self.shrinkage_threshold(1.0/mu2, Z+Lambda2/mu2)
                np.fill_diagonal(C,0.0)
                Lambda1 = Lambda1 + mu1 * (Y - P * Z)
                Lambda2 = Lambda2 + mu2 * (Z - C)
                lambda3 = lambda3 + mu2 * (delta.T * Z - 1.0)
                err1 = np.amax(np.absolute(Z-C))
                err2 = self.error_linear_system(P,Z)
                err3 = np.amax(np.absolute(delta.T * Z - 1.0))
                k += 1

        C = np.array(C[:N,:])
        return C
                
    ##  ADMM algorithm without outliers
    #   @param X DxN NumPy array/matrix representing D points in N-dimensional space
    #   @param use_affine whether or not data points come from union of affine subspaces instead of linear subspaces
    #   @param alpha constant used in calculating updates
    #   @param epsilon termination constant
    #   @param max_iter maximum number of iterations
    #   @return sparse coefficient matrix (NumPy array)
    def lasso_admm(self, X, use_affine = False, alpha = 800.0, epsilon = 0.0002, max_iter = 200):
        
        Y = np.matrix(X)
        N = Y.shape[1]
        mu1 = alpha / self.compute_lambda(Y,Y)
        mu2 = alpha
        C = np.matlib.zeros((N,N))
        
        if not use_affine:
            
            # initializations
            k = 1
            A = np.linalg.pinv(mu1*Y.T*Y + mu2*np.matlib.eye(N))
            Lambda2 = np.matlib.zeros((N,N))
            err1 = 10.0 * epsilon
            
            # main loop
            while k < max_iter and err1 > epsilon:
                Z = A * (mu1*Y.T*Y + mu2*(C-Lambda2/mu2))
                np.fill_diagonal(Z,0.0)
                C = self.shrinkage_threshold(1.0/mu2, Z+Lambda2/mu2)
                np.fill_diagonal(C,0.0)
                Lambda2 = Lambda2 + mu2 * (Z - C)
                err1 = np.amax(np.absolute(Z-C))
                k += 1

        else:
        
            # initializations
            k = 1
            A = np.linalg.pinv(mu1*Y.T*Y + mu2*np.matlib.eye(N) + mu2)
            Lambda2 = np.matlib.zeros((N,N))
            lambda3 = np.matlib.zeros((1,N))
            err1 = 10.0 * epsilon
            err3 = 10.0 * epsilon
            
            # main loop
            while k < max_iter and (err1 > epsilon or err3 > epsilon):
                Z = A * (mu1*Y.T*Y + mu2*(C-Lambda2/mu2) + mu2*np.matlib.ones((N,1))*(1.0-lambda3/mu2))
                np.fill_diagonal(Z,0.0)
                C = self.shrinkage_threshold(1.0/mu2, Z+Lambda2/mu2)
                np.fill_diagonal(C,0.0)
                Lambda2 = Lambda2 + mu2 * (Z - C)
                lambda3 = lambda3 + mu2 * (np.matlib.ones((1,N)) * Z - 1.0)
                err1 = np.amax(np.absolute(Z-C))
                err3 = np.amax(np.absolute(np.matlib.ones((1,N)) * Z - 1.0))
                k += 1

        C = np.array(C)
        return C

    ##  computes sparse coefficient matrix using SSC algorithm with ADMM
    #   @param X NxD NumPy array/matrix representing N points in D-dimensional space
    #   @return sparse coefficient matrix (NumPy array)
    def compute_sparse_coefficient_matrix(self, X):
        XX = np.transpose(X)
        a = self._alpha
        iter = self._max_iter if self._max_iter != -1 else (150 if self._use_outliers else 200)
        C = self.outlier_admm(XX, self._use_affine, a, self._epsilon, iter) if self._use_outliers else self.lasso_admm(XX, self._use_affine, a, self._epsilon, iter)
        return C

    ##  computes predicted labels using SSC algorithm with ADMM
    #   @param inputs array-like matrix, shape (n_samples, n_features) where
    #           n_samples is number of data points and
    #           n_features is ambient dimension
    #   @param timeout This parameter serves as a way for caller to guide the length of the process
    #   @param iterations This parameter serves as a way for caller to guide the length of the process
    #   @return labels A numpy array with the predicted class for each data point
    #   @raise TimeoutError: If time taken for fitting ssc_admm exceeds the timeout parameter
    def produce(self, inputs, timeout=None, iterations=None):
        # type: (Inputs, float, int) -> Outputs
        assert isinstance(inputs, (np.ndarray, np.generic, np.matrix)), "Input should be a numpy array"
        assert inputs.ndim == 2, "Data is not in the right shape"
        assert isinstance(self._k, int), "n_clusters is not integer"

        with stopit.ThreadingTimeout(timeout) as to_ctx_mgr:
            C = self.compute_sparse_coefficient_matrix(inputs)
            W = self.build_adjacency_matrix(C)
            labels = self.spectral_clustering(W, self._k)
            labels = np.array(labels)

        if to_ctx_mgr.state == to_ctx_mgr.EXECUTED:
            return labels
        else:
            raise TimeoutError("SSC ADMM fitting has timed out.")
