# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from .base import *
from .unsupervised_learning import UnsupervisedLearnerPrimitiveBase
__all__ = (u'ClusteringPrimitiveBase',)


class ClusteringPrimitiveBase(UnsupervisedLearnerPrimitiveBase[(Inputs, Outputs, Params)]):
    u'\n    A base class for primitives implementing a clustering algorithm.\n    '
    def fit(self, timeout=None, iterations=None):
        u'\n        A noop.\n        '
        return

    def get_params(self):
        u'\n        A noop.\n        '
        return None

    def set_params(self, params):
        u'\n        A noop.\n        '
        return

    def set_training_data(self, inputs):
        u'\n        A noop.\n        '
        return