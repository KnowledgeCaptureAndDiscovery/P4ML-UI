'''
    spider.tests: cluster_kss_test.py

    Unit tests for the spider.cluster.kss module
'''

import unittest
import tempfile
import os.path
import shutil

import numpy as np
from scipy.linalg import orth
from sklearn.preprocessing import normalize

from spider.cluster.kss import KSS


class testKSS(unittest.TestCase):


    def test_kss(self):
        """ Test runnability and verify shapes of estimated_labels

        """

        # parameters
        D = 100  # dimension of ambient space
        K = 5  # number of subspaces
        Nk = 100  # points per subspace
        d = 1  # dimension of subspace
        varn = 0.01   # noise variance
        N = K * Nk

        # generate data
        X = np.zeros((D, N))
        true_labels = np.zeros(N)
        true_U = np.zeros((K, D, d))
        for kk in range(K):
            true_U[kk] = orth(np.random.randn(D, d))
            x = np.dot(true_U[kk], np.random.randn(d, Nk))
            X[:,range(Nk*kk, Nk*(kk+1))] = x
            true_labels[range(Nk*kk, Nk*(kk+1))] = kk * np.ones(Nk)

        noise = np.sqrt(varn) * np.random.randn(D, N)
        X = X + noise
        X = normalize(X, norm='l2', axis=0)

        # run KSS
        ksub = KSS(n_clusters=K, dim_subspaces=d)
        estimated_labels = ksub.produce(X.T)
        self.assertEqual(len(estimated_labels), N)

if __name__ == '__main__':
    unittest.main()
