import abc

from .base import *

__all__ = ('UnsupervisedLearnerPrimitiveBase',)


class UnsupervisedLearnerPrimitiveBase(PrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    A base class for primitives which have to be fitted before they can start
    producing (useful) outputs from inputs, but they are fitted only on input data.
    """

    @abc.abstractmethod
    def set_training_data(self, *, inputs: Inputs) -> None:  # type: ignore
        """
        Sets training data of this primitive.

        Parameters
        ----------
        inputs : Inputs
            The inputs.
        """
