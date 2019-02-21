import abc
import typing

from d3m_metadata import types

from .base import *
from .transformer import TransformerPrimitiveBase

__all__ = ('PairwiseDistanceLearnerPrimitiveBase', 'PairwiseDistanceTransformerPrimitiveBase', 'InputLabels')

InputLabels = typing.TypeVar('InputLabels', bound=types.Container)


# Defining Generic with all type variables allows us to specify the order and an additional type variable.
class PairwiseDistanceLearnerPrimitiveBase(PrimitiveBase[Inputs, Outputs, Params, Hyperparams], typing.Generic[Inputs, InputLabels, Outputs, Params, Hyperparams]):
    """
    A base class for primitives which learn distances (however defined) between two
    different sets of instances.

    Class is parametrized using five type variables, ``Inputs``, ``InputLabels``, ``Outputs``, ``Params``, and ``Hyperparams``.
    """

    @abc.abstractmethod
    def produce(self, *, inputs: Inputs, second_inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:  # type: ignore
        """
        Computes distance matrix between two sets of data.

        Parameters
        ----------
        inputs : Inputs
            The first set of collections of instances.
        second_inputs : Inputs
            The second set of collections of instances.
        timeout : float
            A maximum time this primitive should take to produce outputs during this method call, in seconds.
        iterations : int
            How many of internal iterations should the primitive do.

        Returns
        ---------
        CallResult[Outputs]
            A n by m distance matrix describing the relationship between each instance in inputs[0] and each instance
            in inputs[1] (n and m are the number of instances in inputs[0] and inputs[1], respectively),
            wrapped inside ``CallResult``.
        """

    @abc.abstractmethod
    def set_training_data(self, *, inputs: Inputs, input_labels: InputLabels) -> None:  # type: ignore
        """
        Sets training data of this primitive.

        Parameters
        ----------
        inputs : Inputs
            The inputs.
        input_labels : InputLabels
            A set of class labels for the inputs.
        """


class PairwiseDistanceTransformerPrimitiveBase(TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    A base class for primitives which compute distances (however defined) between two
    different sets of instances without learning any sort of model.
    """

    @abc.abstractmethod
    def produce(self, *, inputs: Inputs, second_inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:  # type: ignore
        """
        Computes distance matrix between two sets of data.

        Parameters
        ----------
        inputs : Inputs
            The first set of collections of instances.
        second_inputs : Inputs
            The second set of collections of instances.
        timeout : float
            A maximum time this primitive should take to produce outputs during this method call, in seconds.
        iterations : int
            How many of internal iterations should the primitive do.

        Returns
        ---------
        Outputs
            A n by m distance matrix describing the relationship between each instance in inputs[0] and each instance
            in inputs[1] (n and m are the number of instances in inputs[0] and inputs[1], respectively),
            wrapped inside ``CallResult``.
        """
