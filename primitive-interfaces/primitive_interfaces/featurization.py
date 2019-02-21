from .base import *
from .transformer import TransformerPrimitiveBase

__all__ = ('FeaturizationPrimitiveBase', 'FeaturizationTransformerPrimitiveBase')


class FeaturizationPrimitiveBase(PrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    A base class for primitives which transform raw data into a more usable form.

    Use this version for featurizers that allow for fitting (for domain-adaptation, data-specific deep
    learning, etc.).  Otherwise use `FeaturizationTransformerPrimitiveBase`.
    """


class FeaturizationTransformerPrimitiveBase(TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    A base class for primitives which transform raw data into a more usable form.

    Use this version for featurizers that do not require or allow any fitting, and simply
    transform data on demand.  Otherwise use `FeaturizationPrimitiveBase`.
    """
