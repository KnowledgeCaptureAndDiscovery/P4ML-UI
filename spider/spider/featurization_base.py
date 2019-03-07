# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from .base import *
from .transformer import TransformerPrimitiveBase
__all__ = (u'FeaturizationPrimitiveBase',
           u'FeaturizationTransformerPrimitiveBase')


class FeaturizationPrimitiveBase(PrimitiveBase[(Inputs, Outputs, Params)]):
    u'\n    A base class for primitives which transform raw data into a more usable form.\n\n    Use this version for featurizers that allow for fitting (for domain-adaptation, data-specific deep\n    learning, etc.).  Otherwise use `FeaturizationTransformerPrimitiveBase`.\n    '


class FeaturizationTransformerPrimitiveBase(TransformerPrimitiveBase[(Inputs, Outputs)]):
    u'\n    A base class for primitives which transform raw data into a more usable form.\n\n    Use this version for featurizers that do not require or allow any fitting, and simply\n    transform data on demand.  Otherwise use `FeaturizationPrimitiveBase`.\n    '
