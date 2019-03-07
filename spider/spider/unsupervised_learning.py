# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
from .base import *
__all__ = (u'UnsupervisedLearnerPrimitiveBase',)


class UnsupervisedLearnerPrimitiveBase(PrimitiveBase[(Inputs, Outputs, Params)]):
    u'\n    A base class for primitives which have to be fitted before they can start\n    producing (useful) outputs from inputs, but they are fitted only on input data.\n    '

    @abc.abstractmethod
    def set_training_data(self, inputs):
        u'\n        Sets training data of this primitive.\n\n        Parameters\n        ----------\n        inputs : Inputs\n            The inputs.\n        '
