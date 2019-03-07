# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from six import with_metaclass as _py_backwards_six_withmetaclass
import abc
from typing import *
import random
import sys
__all__ = ('Inputs', 'Outputs', 'Params', 'CallMetadata', 'TimeoutError', 'PrimitiveBase', 'ContinueFitMixin', 'SamplingCompositionalityMixin',
           'ProbabilisticCompositionalityMixin', 'Scores', 'Gradients', 'GradientCompositionalityMixin', 'InspectLossMixin')
Inputs = TypeVar('Inputs', bound=Sequence)
Outputs = TypeVar('Outputs', bound=Sequence)
Params = TypeVar('Params')
CallMetadata = NamedTuple(
    'CallMetadata', [('has_finished', bool), ('iterations_done', Optional[int])])
if (sys.version_info[0] == 2):

    class TimeoutError(OSError):
        '\n        Timeout expired.\n        '
else:
    try:
        import builtins
    except ImportError:
        import six.moves.builtins as builtins
    TimeoutError = builtins.TimeoutError


class PrimitiveBase(Generic[(Inputs, Outputs, Params)]):
    '\n    A base class for all TA1 primitives.\n\n    Class is parametrized using three type variables, ``Inputs``, ``Outputs``, and ``Params``.\n    ``Params`` has to be a subclass of a `NamedTuple` and subclasses of this class should\n    define types for all fields of a provided named tuple.`\n\n    All arguments to all methods are keyword-only. In Python 3 this is enforced, in Python 2\n    this is not, but callers should still use only keyword-based arguments when calling to\n    be backwards and future compatible.\n\n    Methods not part of standardized interface classes should be seen as private.\n    Standardized interface does not use attributes so all attributes on classes are seen\n    as private as well. Consider using the convention that private symbols should start with ``_``.\n\n    Method arguments which start with ``_`` are seen as private and can be used for arguments\n    useful for debugging and testing, but they should not be used by (or even known to) a\n    caller during normal execution. Such arguments have to be optional (have a default value)\n    so that the method can be called without the knowledge of the argument.\n\n    Subclasses of this class allow functional compositionality.\n    '
    __metadata__ = {}  # type: Dict[str, Any]

    def __init__(self):
        # type: () -> None
        "\n        All primitives should specify all the hyper-parameters that can be set at the class\n        level in their ``__init__`` as explicit typed keyword-only arguments\n        (no ``*args`` or ``**kwargs``).\n\n        Hyper-parameters are those primitive's parameters which are not changing during\n        a life-time of a primitive. Parameters which do are set using the ``set_params`` method.\n        "

    @abc.abstractmethod
    def produce(self, inputs, timeout=None, iterations=None):
        # type: (Inputs, float, int) -> Outputs
        "\n        Produce primitive's best choice of the output for each of the inputs.\n\n        In many cases producing an output is a quick operation in comparison with ``fit``, but not\n        all cases are like that. For example, a primitive can start a potentially long optimization\n        process to compute outputs. ``timeout`` and ``iterations`` can serve as a way for a caller\n        to guide the length of this process.\n\n        Ideally, a primitive should adapt its call to try to produce the best outputs possible\n        inside the time allocated. If this is not possible and the primitive reaches the timeout\n        before producing outputs, it should raise a ``TimeoutError`` exception to signal that the\n        call was unsuccessful in the given time. The state of the primitive after the exception\n        should be as the method call has never happened and primitive should continue to operate\n        normally. The purpose of ``timeout`` is to give opportunity to a primitive to cleanly\n        manage its state instead of interrupting execution from outside. Maintaining stable internal\n        state should have precedence over respecting the ``timeout`` (caller can terminate the\n        misbehaving primitive from outside anyway). If a longer ``timeout`` would produce\n        different outputs, then ``get_call_metadata``'s ``has_finished`` should be set to\n        ``False``.\n\n        Some primitives have internal iterations (for example, optimization iterations).\n        For those, caller can provide how many of primitive's internal iterations\n        should a primitive do before returning outputs. Primitives should make iterations as\n        small as reasonable. If ``iterations`` is ``None``, then there is no limit on\n        how many iterations the primitive should do and primitive should choose the best amount\n        of iterations on its own (potentially controlled through hyper-parameters).\n        If ``iterations`` is a number, a primitive has to do those number of iterations,\n        if possible. ``timeout`` should still be respected and potentially less iterations\n        can be done because of that. Primitives with internal iterations should make\n        ``get_call_metadata`` returns correct values.\n\n        For primitives which do not have internal iterations, any value of ``iterations``\n        means that they should run fully, respecting only ``timeout``.\n\n        Parameters\n        ----------\n        inputs : Inputs\n            The inputs of shape [num_inputs, ...].\n        timeout : float\n            A maximum time this primitive should take to produce outputs during this method call, in seconds.\n        iterations : int\n            How many of internal iterations should the primitive do.\n\n        Returns\n        -------\n        Outputs\n            The outputs of shape [num_inputs, ...].\n        "

    @abc.abstractmethod
    def fit(self, timeout=None, iterations=None):
        # type: (float, int) -> None
        "\n        Fits primitive using inputs and outputs (if any) using currently set training data.\n\n        If ``fit`` has already been called in the past on different training data,\n        this method fits it **again from scratch** using currently set training data.\n\n        On the other hand, caller can call ``fit`` multiple times on the same training data\n        to continue fitting.\n\n        If ``fit`` fully fits using provided training data, there is no point in making further\n        calls to this method with same training data, and in fact further calls can be noops,\n        or a primitive can decide to refit from scratch.\n\n        In the case fitting can continue with same training data (even if it is maybe not reasonable,\n        because the internal metric primitive is using looks like fitting will be degrading), if ``fit``\n        is called again (without setting training data), the primitive has to continue fitting.\n\n        Caller can provide ``timeout`` information to guide the length of the fitting process.\n        Ideally, a primitive should adapt its fitting process to try to do the best fitting possible\n        inside the time allocated. If this is not possible and the primitive reaches the timeout\n        before fitting, it should raise a ``TimeoutError`` exception to signal that fitting was\n        unsuccessful in the given time. The state of the primitive after the exception should be\n        as the method call has never happened and primitive should continue to operate normally.\n        The purpose of ``timeout`` is to give opportunity to a primitive to cleanly manage\n        its state instead of interrupting execution from outside. Maintaining stable internal state\n        should have precedence over respecting the ``timeout`` (caller can terminate the misbehaving\n        primitive from outside anyway). If a longer ``timeout`` would produce different fitting,\n        then ``get_call_metadata``'s ``has_finished`` should be set to ``False``.\n\n        Some primitives have internal fitting iterations (for example, epochs). For those, caller\n        can provide how many of primitive's internal iterations should a primitive do before returning.\n        Primitives should make iterations as small as reasonable. If ``iterations`` is ``None``,\n        then there is no limit on how many iterations the primitive should do and primitive should\n        choose the best amount of iterations on its own (potentially controlled through\n        hyper-parameters). If ``iterations`` is a number, a primitive has to do those number of\n        iterations (even if not reasonable), if possible. ``timeout`` should still be respected\n        and potentially less iterations can be done because of that. Primitives with internal\n        iterations should make ``get_call_metadata`` returns correct values.\n\n        For primitives which do not have internal iterations, any value of ``iterations``\n        means that they should fit fully, respecting only ``timeout``.\n\n        Subclasses can extend arguments of this method with explicit typed keyword arguments used during\n        the fitting process. For example, they can accept other primitives through an argument representing\n        a regularizer to use during fitting. The reason why those are not part of constructor arguments is\n        that one can create primitives in any order before having to invoke them or pass them to other\n        primitives.\n\n        Parameters\n        ----------\n        timeout : float\n            A maximum time this primitive should be fitting during this method call, in seconds.\n        iterations : int\n            How many of internal iterations should the primitive do.\n        "

    @abc.abstractmethod
    def get_params(self):
        # type: () -> Params
        '\n        Returns parameters of this primitive.\n\n        Parameters are all parameters of the primitive which can potentially change during a life-time of\n        a primitive. Parameters which cannot are passed through constructor.\n\n        Parameters should include all data which is necessary to create a new instance of this primitive\n        behaving exactly the same as this instance, when the new instance is created by passing the same\n        parameters to the class constructor and calling ``set_params``.\n\n        Returns\n        -------\n        Params\n            A named tuple of parameters.\n        '

    @abc.abstractmethod
    def set_params(self, params):
        # type: (Params) -> None
        '\n        Sets parameters of this primitive.\n\n        Parameters are all parameters of the primitive which can potentially change during a life-time of\n        a primitive. Parameters which cannot are passed through constructor.\n\n        Parameters\n        ----------\n        params : Params\n            A named tuple of parameters.\n        '

    def get_call_metadata(self):
        # type: () -> CallMetadata
        '\n        Returns metadata about the last ``produce`` or ``fit`` call.\n\n        For ``produce``, ``has_finished`` is ``True`` if the last call to ``produce``\n        has produced the final outputs and a call with more time or more iterations\n        cannot get different outputs.\n\n        For ``fit``, ``has_finished`` is ``True`` if a primitive has been fully fitted\n        on current training data and further calls to ``fit`` are unnecessary and will\n        not change anything. ``False`` means that more iterations can be done (but it\n        does not necessary mean that more iterations are beneficial).\n\n        If a primitive has iterations internally, then ``iterations_done`` contains\n        how many of those iterations have been made during the last call. If primitive\n        does not support them, ``iterations_done`` is ``None``.\n\n        The reason why this is a separate call is to make return value from ``produce`` and\n        ``fit`` simpler. Moreover, not all callers might care about this information and for\n        many primitives a default implementation of this method works.\n\n        Returns\n        -------\n        CallMetadata\n            A named tuple with metadata.\n        '
        return CallMetadata(has_finished=True, iterations_done=None)

    def set_random_seed(self, seed):
        # type: (int) -> None
        "\n        Sets a random seed for all operations from now on inside the primitive.\n\n        By default it sets numpy's and Python's random seed.\n\n        Parameters\n        ----------\n        seed : int\n            A random seed to use.\n        "
        try:
            import numpy
            numpy.random.seed(seed)
        except ImportError:
            pass
        random.seed(seed)


class ContinueFitMixin(Generic[(Inputs, Outputs, Params)]):

    @abc.abstractmethod
    def continue_fit(self, timeout=None, iterations=None):
        # type: (float, int) -> None
        '\n        Similar to base ``fit``, this method fits the primitive using inputs and outputs (if any)\n        using currently set training data.\n\n        The difference is what happens when currently set training data is different from\n        what the primitive might have already been fitted on. ``fit`` fits the primitive from\n        scratch, while ``continue_fit`` fits it further and does **not** start from scratch.\n\n        Caller can still call ``continue_fit`` multiple times on the same training data as well,\n        in which case primitive should try to improve the fit in the same way as with ``fit``.\n\n        From the perspective of a caller of all other methods, the training data in effect\n        is still just currently set training data. If a caller wants to call ``gradient_output``\n        on all data on which the primitive has been fitted through multiple calls of ``continue_fit``\n        on different training data, the caller should pass all this data themselves through\n        another call to ``set_training_data``, do not call ``fit`` or ``continue_fit`` again,\n        and use ``gradient_output`` method. In this way primitives which truly support\n        continuation of fitting and need only the latest data to do another fitting, do not\n        have to keep all past training data around themselves.\n\n        If a primitive supports this mixin, then both ``fit`` and ``continue_fit`` can be\n        called. ``continue_fit`` always continues fitting, if it was started through ``fit``\n        or ``continue_fit``. And ``fit`` always restarts fitting, even if previously\n        ``continue_fit`` was used.\n\n        When this mixin is supported, then ``get_call_metadata`` method should return\n        metadata also for call of ``continue_fit``.\n\n        Parameters\n        ----------\n        timeout : float\n            A maximum time this primitive should be fitting during this method call, in seconds.\n        iterations : int\n            How many of internal iterations should the primitive do.\n        '


class SamplingCompositionalityMixin(Generic[(Inputs, Outputs, Params)]):
    '\n    This mixin signals to a caller that the primitive is probabilistic but\n    may be likelihood free.\n\n    Mixin should be used together with the ``PrimitiveBase`` class.\n    '

    @abc.abstractmethod
    def sample(self, inputs, num_samples=1, timeout=None, iterations=None):
        # type: (Inputs, int, float, int) -> Sequence[Outputs]
        '\n        Sample each input from ``inputs`` ``num_samples`` times.\n\n        Semantics of ``timeout`` and ``iterations`` is the same as in ``produce``.\n\n        When this mixin is supported, then ``get_call_metadata`` method should return\n        metadata also for call of ``sample``.\n\n        Parameters\n        ----------\n        inputs : Inputs\n            The inputs of shape [num_inputs, ...].\n        num_samples : int\n            The number of samples to return in a set of samples.\n        timeout : float\n            A maximum time this primitive should take to sample outputs during this method call, in seconds.\n        iterations : int\n            How many of internal iterations should the primitive do.\n\n        Returns\n        -------\n        Sequence[Outputs]\n            The multiple sets of samples of shape [num_samples, num_inputs, ...].\n        '


class ProbabilisticCompositionalityMixin(Generic[(Inputs, Outputs, Params)]):
    '\n    This mixin provides additional abstract methods which primitives should implement to\n    help callers with doing various end-to-end refinements using probabilistic\n    compositionality.\n\n    This mixin adds methods to support at least:\n\n    * Metropolis-Hastings\n\n    Mixin should be used together with the ``PrimitiveBase`` class and ``SamplingCompositionalityMixin`` mixin.\n    '

    @abc.abstractmethod
    def log_likelihood(self, outputs, inputs):
        # type: (Outputs, Inputs) -> float
        '\n        Returns log probability of outputs given inputs and params under this primitive:\n\n        sum_i(log(p(output_i | input_i, params)))\n\n        Parameters\n        ----------\n        outputs : Outputs\n            The outputs.\n        inputs : Inputs\n            The inputs.\n\n        Returns\n        -------\n        float\n            sum_i(log(p(output_i | input_i, params)))\n        '


class Scores(Generic[Params]):
    '\n    A type representing a named tuple which holds all the differentiable fields from ``Params``.\n    Their values are of type ``float``.\n    '


class Gradients(Generic[Outputs]):
    '\n    A type representing a structure of one sample from ``Outputs``, but the values are of type\n    ``Optional[float]``. Value is ``None`` if gradient for that part of the structure is not possible.\n    '


class GradientCompositionalityMixin(Generic[(Inputs, Outputs, Params)]):
    '\n    This mixin provides additional abstract methods which primitives should implement to\n    help callers with doing various end-to-end refinements using gradient-based\n    compositionality.\n\n    This mixin adds methods to support at least:\n\n    * gradient-based, compositional end-to-end training\n    * regularized pre-training\n    * multi-task adaptation\n    * black box variational inference\n    * Hamiltonian Monte Carlo\n    '

    @abc.abstractmethod
    def gradient_output(self, outputs, inputs):
        # type: (Outputs, Inputs) -> Gradients[Outputs]
        '\n        Returns the gradient of loss sum_i(L(output_i, produce_one(input_i))) with respect to output.\n\n        When fit term temperature is set to non-zero, it should return the gradient with respect to output of:\n\n        sum_i(L(output_i, produce_one(input_i))) + temperature * sum_i(L(training_output_i, produce_one(training_input_i)))\n\n        When used in combination with the ``ProbabilisticCompositionalityMixin``, it returns gradient\n        of sum_i(log(p(output_i | input_i, params))) with respect to output.\n\n        When fit term temperature is set to non-zero, it should return the gradient with respect to output of:\n\n        sum_i(log(p(output_i | input_i, params))) + temperature * sum_i(log(p(training_output_i | training_input_i, params)))\n\n        Parameters\n        ----------\n        outputs : Outputs\n            The outputs.\n        inputs : Inputs\n            The inputs.\n\n        Returns\n        -------\n        Gradients[Outputs]\n            Gradients.\n        '

    @abc.abstractmethod
    def gradient_params(self, outputs, inputs):
        # type: (Outputs, Inputs) -> Scores[Params]
        '\n        Returns the gradient of loss sum_i(L(output_i, produce_one(input_i))) with respect to params.\n\n        When fit term temperature is set to non-zero, it should return the gradient with respect to params of:\n\n        sum_i(L(output_i, produce_one(input_i))) + temperature * sum_i(L(training_output_i, produce_one(training_input_i)))\n\n        When used in combination with the ``ProbabilisticCompositionalityMixin``, it returns gradient of\n        log(p(output | input, params)) with respect to params.\n\n        When fit term temperature is set to non-zero, it should return the gradient with respect to params of:\n\n        sum_i(log(p(output_i | input_i, params))) + temperature * sum_i(log(p(training_output_i | training_input_i, params)))\n\n        Parameters\n        ----------\n        outputs : Outputs\n            The outputs.\n        inputs : Inputs\n            The inputs.\n\n        Returns\n        -------\n        Scores[Params]\n            A named tuple with all fields from ``Params`` and values set to gradient for each parameter.\n        '

    @abc.abstractmethod
    def set_fit_term_temperature(self, temperature=0):
        # type: (float) -> None
        '\n        Sets the temperature used in ``gradient_output`` and ``gradient_params``.\n\n        Parameters\n        ----------\n        temperature : float\n            The temperature to use, [0, inf), typically, [0, 1].\n        '


class InspectLossMixin(
        _py_backwards_six_withmetaclass(abc.ABCMeta, *[object])):
    '\n    Mixin which provides an abstract method for a caller to call to inspect which\n    loss function a primitive is using internally.\n    '

    @abc.abstractmethod
    def get_loss_function(self):
        # type: () -> Optional[str]
        '\n        Returns a D3M standard name of the loss function used by the primitive, or ``None`` if using\n        a non-standard loss function or if the primitive does not use a loss function at all.\n\n        Returns\n        -------\n        str\n            A D3M standard name of the loss function used.\n        '

