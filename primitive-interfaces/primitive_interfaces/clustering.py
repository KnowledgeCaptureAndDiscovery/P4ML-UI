import abc
import typing

from d3m_metadata import types

from .base import *
from .unsupervised_learning import UnsupervisedLearnerPrimitiveBase

__all__ = ('ClusteringPrimitiveBase', 'DistanceMatrixOutput', 'ClusteringDistanceMatrixMixin')

DistanceMatrixOutput = typing.TypeVar('DistanceMatrixOutput', bound=types.Container)


class ClusteringPrimitiveBase(UnsupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    A base class for primitives implementing a clustering algorithm.
    """

    @abc.abstractmethod
    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        ``produce`` method should return a membership map.

        A data structure that for each input sample tells to which cluster that sample was assigned to. So ``Outputs``
        should have the same number of samples than ``Inputs``, and the value at each output sample should represent
        a cluster. Consider representing it with just a simple numeric identifier.

        Parameters
        ----------
        inputs : Inputs
            The inputs of shape [num_inputs, ...].
        timeout : float
            A maximum time this primitive should take to produce outputs during this method call, in seconds.
        iterations : int
            How many of internal iterations should the primitive do.

        Returns
        -------
        CallResult[Outputs]
            The outputs of shape [num_inputs, 1] wrapped inside ``CallResult`` for a simple numeric
            cluster identifier.
        """


class ClusteringDistanceMatrixMixin(typing.Generic[Inputs, Outputs, Params, Hyperparams, DistanceMatrixOutput]):
    @abc.abstractmethod
    def produce_distance_matrix(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[DistanceMatrixOutput]:
        """
        Semantics of this call are the same as the call to a regular ``produce`` method, just
        that the output is a distance matrix instead of a membership map.

        Parameters
        ----------
        inputs : Inputs
            The inputs of shape [num_inputs, ...].
        timeout : float
            A maximum time this primitive should take to produce outputs during this method call, in seconds.
        iterations : int
            How many of internal iterations should the primitive do.

        Returns
        -------
        CallResult[DistanceMatrixOutput]
            The distance matrix of shape [num_inputs, num_inputs, ...] wrapped inside ``CallResult``, where (i, j) element
            of the matrix represent a distance between i-th and j-th sample in the inputs.
        """
