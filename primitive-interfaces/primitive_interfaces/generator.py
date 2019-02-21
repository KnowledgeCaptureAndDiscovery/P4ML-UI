import abc

from d3m_metadata import container

from .base import *

__all__ = ('GeneratorPrimitiveBase',)


class GeneratorPrimitiveBase(PrimitiveBase[container.List[None], Outputs, Params, Hyperparams]):
    """
    A base class for primitives which have to be fitted before they can start
    producing (useful) outputs, but they are fitted only on output data.
    Moreover, they do not accept any inputs to generate outputs, which is
    represented as a list of ``None`` values to ``produce`` method to signal
    how many outputs are requested.

    This class is parametrized using only by three type variables,
    ``Outputs``, ``Params``, and ``Hyperparams``.
    """

    @abc.abstractmethod
    def set_training_data(self, *, outputs: Outputs) -> None:  # type: ignore
        """
        Sets training data of this primitive.

        Parameters
        ----------
        outputs : Outputs
            The outputs.
        """
