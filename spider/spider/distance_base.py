# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
from typing import *
from .base import *
from .transformer import TransformerPrimitiveBase
__all__ = ('PairwiseDistanceLearnerPrimitiveBase',
           'PairwiseDistanceTransformerPrimitiveBase', 'InputLabels')
InputLabels = TypeVar('InputLabels', bound=Sequence)


class PairwiseDistanceLearnerPrimitiveBase(PrimitiveBase[(Tuple[(Inputs, Inputs)], Outputs, Params)], Generic[(Inputs, InputLabels, Outputs, Params)]):
    '\n    A base class for primitives which learn distances (however defined) between two\n    different sets of instances.\n\n    Class is parametrized using four type variables, ``Inputs``, ``InputLabels``, ``Outputs``, and ``Params``.\n    '

    @abc.abstractmethod
    def produce(self, inputs, timeout=None, iterations=None):
        # type: (Tuple[(Inputs, Inputs)], float, int) -> Outputs
        '\n        Computes distance matrix between two sets of data.\n\n        Parameters\n        ----------\n        inputs : Tuple[Inputs, Inputs]\n            A pair of collections of instances.\n        timeout : float\n            A maximum time this primitive should take to produce outputs during this method call, in seconds.\n        iterations : int\n            How many of internal iterations should the primitive do.\n\n        Returns\n        ---------\n        Outputs\n            A n by m distance matrix describing the relationship between each instance in inputs[0] and each instance\n            in inputs[1] (n and m are the number of instances in inputs[0] and inputs[1], respectively).\n        '

    @abc.abstractmethod
    def set_training_data(self, inputs, outputs):
        # type: (Inputs, InputLabels) -> None
        '\n        Sets training data of this primitive.\n\n        Parameters\n        ----------\n        inputs : Inputs\n            The inputs.\n        outputs : InputLabels\n            A set of class labels for the inputs.\n        '


class PairwiseDistanceTransformerPrimitiveBase(TransformerPrimitiveBase[(Tuple[(Inputs, Inputs)], Outputs)]):
    '\n    A base class for primitives which compute distances (however defined) between two\n    different sets of instances without learning any sort of model.\n    '

    @abc.abstractmethod
    def produce(self, inputs, timeout=None, iterations=None):
        # type: (Tuple[(Inputs, Inputs)], float, int) -> Outputs
        '\n        Computes distance matrix between two sets of data.\n\n        Parameters\n        ----------\n        inputs : Tuple[Inputs, Inputs]\n            A pair of collections of instances.\n        timeout : float\n            A maximum time this primitive should take to produce outputs during this method call, in seconds.\n        iterations : int\n            How many of internal iterations should the primitive do.\n\n        Returns\n        ---------\n        Outputs\n            A n by m distance matrix describing the relationship between each instance in inputs[0] and each instance\n            in inputs[1] (n and m are the number of instances in inputs[0] and inputs[1], respectively).\n        '
