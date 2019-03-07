from __future__ import print_function

import warnings
import pickle
import stopit
import copy
import numpy as np

from typing import *
from exceptions import ValueError, TypeError
from sklearn.ensemble import RandomForestRegressor
from ...distance_base import PairwiseDistanceLearnerPrimitiveBase, TimeoutError
from spider.distance.utils import get_random_constraints

Inputs = np.ndarray
InputLabels = np.ndarray
Outputs = np.ndarray
Params = tuple

class RFD(PairwiseDistanceLearnerPrimitiveBase[Inputs, InputLabels, Outputs, Params]):

    def __init__(self, class_cons=1000,
                 num_trees=500,
                 min_node_size=1,
                 n_jobs=-1,
                 verbose=0):
        # type: (int, int, int, int, int) -> None
        """
        Primitive for learning Random Forest Distance metrics and returning a pairwise
        distance matrix between two sets of data.

        Arguments:
            class_cons: the number of pairwise constraints per class
                to sample from the training labels
            num_trees: the number of trees the metric forest should contain
                (default 500)
            min_node_size: the stopping criterion for tree splitting. Trees will
                be split until each leaf node is this size or smaller (default
                1)
            n_jobs: the number of separate processes that should be used in 
                training and inference.  If -1, then n_jobs is set equal to the 
                number of cores (default -1)
            verbose: controls verbosity of the tree-building process (default 0)

        Returns:
            None

        Raises:
            None
        """

        super(RFD, self).__init__()

        self.fitted = False
        self.class_cons = class_cons

        self.rf = RandomForestRegressor(n_estimators=num_trees, max_features="log2",
                                        min_samples_leaf=min_node_size, n_jobs=n_jobs,
                                        verbose=verbose)

    def set_training_data(self, inputs, outputs):
        # type: (Inputs, InputLabels) -> None
        self.X = inputs
        self.y = outputs
        self.fitted = False


    def fit(self, timeout=None, iterations=None):
        # type: (float, int) -> None
        """
        Fit the random forest distance to a set of labeled data by sampling and fitting
        to pairwise constraints.
        """

        # state/input checking
        if self.fitted:
            return

        if not hasattr(self, 'X') or not hasattr(self, 'y'):
            raise ValueError("Missing training data.")

        if(not (isinstance(self.X, np.ndarray) and isinstance(self.y, np.ndarray))):
            raise TypeError('Training inputs and outputs must be numpy arrays.')

        # store state in case of timeout
        if hasattr(self, 'd'):
            dtemp = self.d
        else:
           dtemp = None
        rftemp = copy.deepcopy(self.rf)

        # do fitting with timeout
        with stopit.ThreadingTimeout(timeout) as to_ctx_mgr:
            n = self.X.shape[0]
            self.d = self.X.shape[1]
            assert n > 0
            assert self.d > 0
            assert n == self.y.shape[0]

            constraints = get_random_constraints(
                self.y, int(self.class_cons / 3), 2 * int(self.class_cons / 3))

            c1 = self.X[constraints[:, 0], :]
            c2 = self.X[constraints[:, 1], :]
            rfdfeat = np.empty(dtype=np.float32, shape=(constraints.shape[0], self.d * 2))
            rfdfeat[:, :self.d] = np.abs(c1 - c2)
            rfdfeat[:, self.d:] = (c1 + c2) / 2

            self.rf.fit(rfdfeat, constraints[:, 2])
            self.fitted = True

        # if we completed on time, return.  Otherwise reset state and raise error.
        if to_ctx_mgr.state == to_ctx_mgr.EXECUTED:
            return
        else:
            self.d = dtemp
            self.rf = rftemp
            self.fitted = False
            raise TimeoutError("RFD fitting timed out.")
            

    def get_params(self):
        # type: () -> Params
        return Params((self.fitted, self.d, pickle.dumps(self.rf)))


    def set_params(self, params):
        # type: (Params) -> None
        self.fitted = params[0]
        self.d = params[1]
        self.rf = pickle.loads(params[2])


    def produce(self, inputs, timeout=None, iterations=None):
        # type: (Tuple[(Inputs, Inputs)], float, int) -> Outputs
        """
        Compute the distance matrix between vector arrays inputs[0] and
        inputs[1], yielding an output of shape n by m (where n and m are
        the number of instances in inputs[0] and inputs[1] respectively).

        Both inputs must match the dimensionality of the training data.
        The same array may be input twice in order to generate an
        (inverted) kernel matrix for use in clustering, etc.

        Raises:
            AssertError: thrown if the model has not yet been fit to data
            TypeError: thrown if X or Y are not both numpy arrays
            ValueError: thrown if the dimensionality of instance(s) in
                X or Y is not equal to the dimensionality of the
                training data (or the matrices are otherwise the wrong
                shape)
        """
        # first do assorted error checking and initialization
        assert self.fitted == True

        X = inputs[0]
        Y = inputs[1]

        if(not (isinstance(X, np.ndarray) and (isinstance(Y, np.ndarray)))):
            raise TypeError('Both inputs must be numpy arrays.')

        if(X.shape[1] != self.d or Y.shape[1] != self.d):
            raise ValueError('Input has the wrong dimensionality.')

        n1 = X.shape[0]
        n2 = Y.shape[0]

        # start timeout counter
        with stopit.ThreadingTimeout(timeout) as to_ctx_mgr:
            # compute distance from each instance in X to all instances in Y
            dist = np.empty(dtype=np.float32, shape=(n1, n2))
            for i in xrange(0, n1):
                data = np.empty(dtype=np.float32, shape=(n2, self.d * 2))
                data[:, :self.d] = np.abs(X[i, :] - Y)
                data[:, self.d:] = (X[i, :] + Y) / 2
                dist[i, :] = self.rf.predict(data)

            # return distance
            return dist
        # if we did not finish in time, raise error.
        if to_ctx_mgr.state != to_ctx_mgr.EXECUTED:
            raise TimeoutError("RFD produce timed out.")
