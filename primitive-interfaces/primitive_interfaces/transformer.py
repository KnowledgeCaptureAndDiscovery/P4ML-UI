from .base import *

__all__ = ('TransformerPrimitiveBase',)


class TransformerPrimitiveBase(PrimitiveBase[Inputs, Outputs, None, Hyperparams]):
    """
    A base class for primitives which are not fitted at all and can
    simply produce (useful) outputs from inputs directly. As such they
    also do not have any state (params).

    This class is parametrized using only three type variables, ``Inputs``,
    ``Outputs``, and ``Hyperparams``.
    """

    def set_training_data(self) -> None:  # type: ignore
        """
        A noop.
        """

        return

    def fit(self, *, timeout: float = None, iterations: int = None) -> None:
        """
        A noop.
        """

        return

    def get_params(self) -> None:
        """
        A noop.
        """

        return None

    def set_params(self, *, params: None) -> None:
        """
        A noop.
        """

        return
