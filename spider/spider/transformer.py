# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from .base import *
__all__ = (u'TransformerPrimitiveBase',)


class TransformerPrimitiveBase(PrimitiveBase[(Inputs, Outputs, None)]):
    u'\n    A base class for primitives which are not fitted at all and can\n    simply produce (useful) outputs from inputs directly. As such they\n    also do not have any state (params).\n\n    This class is parametrized using only two type variables, ``Inputs`` and ``Outputs``.\n    '

    def fit(self, timeout=None, iterations=None):
        u'\n        A noop.\n        '
        return

    def get_params(self):
        u'\n        A noop.\n        '
        return None

    def set_params(self, params):
        u'\n        A noop.\n        '
        return
