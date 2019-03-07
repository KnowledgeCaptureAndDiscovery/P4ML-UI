import abc
import inspect
import typing

from d3m_metadata import hyperparams, metadata as metadata_module, params, problem, types, utils

__all__ = (
    'Inputs', 'Outputs', 'Params', 'Hyperparams', 'CallResult',
    'PrimitiveBase', 'ContinueFitMixin', 'SamplingCompositionalityMixin',
    'ProbabilisticCompositionalityMixin', 'Scores', 'Gradients',
    'GradientCompositionalityMixin', 'LossFunctionMixin',
)


Inputs = typing.TypeVar('Inputs', bound=types.Container)
Outputs = typing.TypeVar('Outputs', bound=types.Container)
# This type parameter is optional and can be set to None.
# See "TransformerPrimitiveBase" for an example.
Params = typing.TypeVar('Params', bound=params.Params)
Hyperparams = typing.TypeVar('Hyperparams', bound=hyperparams.Hyperparams)

T = typing.TypeVar('T')


class CallResult(typing.Generic[T]):
    """
    Some methods return additional metadata about the method call itself
    (which is different to metadata about the value returned, which is stored
    in ``metadata`` attribute of the value itself).

    For ``produce`` method call, ``has_finished`` is ``True`` if the last call
    to ``produce`` has produced the final outputs and a call with more time or
    more iterations cannot get different outputs.

    For ``fit`` method call, ``has_finished`` is ``True`` if a primitive has been
    fully fitted on current training data and further calls to ``fit`` are
    unnecessary and will not change anything. ``False`` means that more iterations
    can be done (but it does not necessary mean that more iterations are beneficial).

    If a primitive has iterations internally, then ``iterations_done`` contains
    how many of those iterations have been made during the last call. If primitive
    does not support them, ``iterations_done`` is ``None``.

    Those methods should return value wrapped into this class.

    Parameters
    ----------
    value : Any
        The value itself of the method call.
    has_finished : bool
        Set to ``True`` if it is not reasonable to call the method again anymore.
    iterations_done : int
        How many iterations have been done during a method call, if any.
    """

    def __init__(self, value: T, has_finished: bool = True, iterations_done: int = None) -> None:
        self.value = value
        self.has_finished = has_finished
        self.iterations_done = iterations_done


class PrimitiveBaseMeta(typing.GenericMeta):
    """
    A metaclass which provides the primitive instance to metadata so that primitive
    metadata can be automatically generated.
    """

    def __new__(mcls, class_name, bases, namespace, **kwargs):  # type: ignore
        cls = super().__new__(mcls, class_name, bases, namespace, **kwargs)

        if inspect.isabstract(cls):
            return cls

        if not isinstance(cls.metadata, metadata_module.PrimitiveMetadata):
            raise TypeError("'metadata' attribute is not an instance of PrimitiveMetadata.")

        cls.metadata.contribute_to_class(cls)

        return cls


class PrimitiveBase(typing.Generic[Inputs, Outputs, Params, Hyperparams], metaclass=PrimitiveBaseMeta):
    """
    A base class for all TA1 primitives.

    Class is parametrized using four type variables, ``Inputs``, ``Outputs``, ``Params``,
    and ``Hyperparams``.

    ``Params`` has to be a subclass of `d3m_metadata.params.Params` and should define
    all fields and their types for parameters which the primitive is fitting.

    ``Hyperparams`` has to be a subclass of a `d3m_metadata.hyperparams.Hyperparams`.
    Hyper-parameters are those primitive's parameters which primitive is not fitting and
    generally do not change during a life-time of a primitive.

    ``Params`` and ``Hyperparams`` have to be pickable and copyable. See `pickle`,
    `copy`, and `copyreg` Python modules for more information.

    In this context we use term method arguments to mean both formal parameters and
    actual parameters of a method. We do this to not confuse method parameters with
    primitive parameters (``Params``).

    All arguments to all methods are keyword-only. No ``*args`` or ``**kwargs`` should
    ever be used in any method.

    Standardized interface use few public attributes and no other public attributes are
    allowed to assure future compatibility. For your attributes use the convention that
    private symbols should start with ``_``.

    Primitives can have methods which are not part of standardized interface classes:

    * Additional "produce" methods which are prefixed with ``produce_`` and have
      the same semantics as ``produce`` but potentially return different output
      container types instead of ``Outputs`` (in such primitive ``Outputs`` is seen as
      primary output type, but the primitive also has secondary output types).
      They should return ``CallResult`` and have ``timeout`` and ``iterations`` arguments.
    * Private methods prefixed with ``_``.

    No other public additional methods are allowed. If this represents a problem for you,
    open an issue. (The rationale is that for other methods an automatic system will not
    understand the semantics of the method.)

    Method arguments which start with ``_`` are seen as private and can be used for arguments
    useful for debugging and testing, but they should not be used by (or even known to) a
    caller during normal execution. Such arguments have to be optional (have a default value)
    so that the method can be called without the knowledge of the argument.

    All arguments to all methods together are seen as arguments to the primitive as a whole.
    They are identified by their names. This means that any argument name must have the same
    type and semantics across all methods, effectively be the same argument. In addition,
    all hyper-parameters can also be overridden for a method call. To allow for this, methods
    can accept arguments which match hyper-parameter names. All this is necessary so that callers
    can have easier time determine what values to pass to arguments and that it is
    easier to describe what all values are inputs to a primitive as a whole (set of all
    arguments).

    To recap, subclasses can extend arguments of standard methods with explicit typed keyword
    arguments used for the method call, or define new "produce" methods with arbitrary explicit
    typed keyword. There are multiple kinds of such arguments allowed:

    * An argument which is overriding a hyper-parameter for the duration of the call.
      It should match a hyper-parameter in name and type. It should be a required argument
      (no default value) which the caller has to supply (e.g., or with a default value of a
      hyper-parameter, or with the same hyper-parameter as it was passed to the constructor,
      or with some other value).
    * An (additional) input argument of any container type and not necessary of ``Inputs``
      (in such primitive ``Inputs`` is seen as primary input type, but the primitive also has
      secondary input types).
    * A primitive. In this case a caller will pass in an instance of a primitive (potentially
      connecting it to other primitives and fitting it, if necessary).
    * An (additional) value argument which is one of standard data types, but not a container type.
      In this case a caller will try to satisfy the input by creating part of a pipeline which
      ends with a singleton primitive and extract the singleton value and pass it without a container.
      This kind of an argument is **discouraged** and should probably be a hyper-parameter instead
      (because it is unclear how can a caller determine which value is a reasonable value to pass
      in an automatic way), but it is defined for completeness and so that existing pipelines can be
      easier described.
    * A private argument prefixed with ``_`` which is used for debugging and testing.
      It should not be used by (or even known to) a caller during normal execution.
      Such argument has to be optional (have a default value) so that the method can be called
      without the knowledge of the argument.

    Subclasses of this class allow functional compositionality.

    Attributes
    ----------
    metadata : PrimitiveMetadata
        Primitive's metadata. Available as a class attribute.
    hyperparams : Hyperparams
        Hyperparams passed to the constructor.
    random_seed : int
        Random seed passed to the constructor.
    docker_containers : Dict[str, str]
        A dict mapping Docker image keys from primitive's metadata to container addresses
        under which containers are accessible by the primitive.
    """

    # Primitive's metadata (annotation) should be put on "metadata' attribute to provide
    # all fields (which cannot be determined automatically) inside the code. In this way metadata
    # is close to the code and it is easier for consumers to make sure metadata they are using
    # is really matching the code they are using. PrimitiveMetadata class will automatically
    # extract additional metadata and update itself with metadata about code and other things
    # it can extract automatically.
    metadata: metadata_module.PrimitiveMetadata = None

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, docker_containers: typing.Dict[str, str] = None) -> None:
        """
        All primitives should accept all their hyper-parameters in a constructor as one value,
        an instance of type ``Hyperparams``.

        Methods can accept per-call overrides for some hyper-parameters (e.g., a threshold
        for fitting). Those arguments have to match in name and type a hyper-parameter defined
        in ``hyperparams`` object. The value provided in the ``hyperparams`` object serves
        as a default in such case.

        Provided random seed should control all randomness used by this primitive.
        Primitive should behave exactly the same for the same random seed across multiple
        invocations. You can call `numpy.random.RandomState(random_seed)` to obtain an
        instance of a random generator using provided seed.

        Primitives can be wrappers around or use one or more Docker images which they can
        specify as part of  ``installation`` field in their metadata. Each Docker image listed
        there has a ``key`` field identifying that image. When primitive is created,
        ``docker_containers`` contains a mapping between those keys and addresses to which
        primitive can connect to access a running Docker container for a particular Docker
        image. Docker containers might be long running and shared between multiple instances
        of a primitive.

        No other arguments to the constructor are allowed (except for private arguments)
        because we want instances of primitives to be created without a need for any other
        prior computation.

        Constructor should be kept lightweight and not do any computation. No resources
        should be allocated in the constructor. Resources should be allocated when needed.
        """

        self.hyperparams = hyperparams
        self.random_seed = random_seed
        self.docker_containers = docker_containers

    @abc.abstractmethod
    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Produce primitive's best choice of the output for each of the inputs.

        The output value should be wrapped inside ``CallResult`` object before returning.

        In many cases producing an output is a quick operation in comparison with ``fit``, but not
        all cases are like that. For example, a primitive can start a potentially long optimization
        process to compute outputs. ``timeout`` and ``iterations`` can serve as a way for a caller
        to guide the length of this process.

        Ideally, a primitive should adapt its call to try to produce the best outputs possible
        inside the time allocated. If this is not possible and the primitive reaches the timeout
        before producing outputs, it should raise a ``TimeoutError`` exception to signal that the
        call was unsuccessful in the given time. The state of the primitive after the exception
        should be as the method call has never happened and primitive should continue to operate
        normally. The purpose of ``timeout`` is to give opportunity to a primitive to cleanly
        manage its state instead of interrupting execution from outside. Maintaining stable internal
        state should have precedence over respecting the ``timeout`` (caller can terminate the
        misbehaving primitive from outside anyway). If a longer ``timeout`` would produce
        different outputs, then ``CallResult``'s ``has_finished`` should be set to ``False``.

        Some primitives have internal iterations (for example, optimization iterations).
        For those, caller can provide how many of primitive's internal iterations
        should a primitive do before returning outputs. Primitives should make iterations as
        small as reasonable. If ``iterations`` is ``None``, then there is no limit on
        how many iterations the primitive should do and primitive should choose the best amount
        of iterations on its own (potentially controlled through hyper-parameters).
        If ``iterations`` is a number, a primitive has to do those number of iterations,
        if possible. ``timeout`` should still be respected and potentially less iterations
        can be done because of that. Primitives with internal iterations should make
        ``CallResult`` contain correct values.

        For primitives which do not have internal iterations, any value of ``iterations``
        means that they should run fully, respecting only ``timeout``.

        Parameters
        ----------
        inputs : Inputs
            The inputs of shape [num_inputs, ...].
        timeout : float
            A maximum time this primitive should take to produce outputs during this method call, in seconds.
        iterations : int
            How many of internal iterations should the primitive do.

        Returns
        -------
        CallResult[Outputs]
            The outputs of shape [num_inputs, ...] wrapped inside ``CallResult``.
        """

    @abc.abstractmethod
    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        """
        Sets training data of this primitive.

        Standard sublasses in this package do not adhere to the Liskov substitution principle when
        inheriting this method because they do not necessary accept all arguments found in the base
        class. This means that one has to inspect which arguments are accepted at runtime, or in
        other words, one has to inspect which exactly subclass a primitive implements, if
        you are accepting a wider range of primitives. This relaxation is allowed only for
        standard subclasses found in this package. Primitives themselves should not break
        the Liskov substitution principle but should inherit from a suitable base class.

        Parameters
        ----------
        inputs : Inputs
            The inputs.
        outputs : Outputs
            The outputs.
        """

    @abc.abstractmethod
    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        """
        Fits primitive using inputs and outputs (if any) using currently set training data.

        The returned value should be a ``CallResult`` object with ``value`` set to ``None``.

        If ``fit`` has already been called in the past on different training data,
        this method fits it **again from scratch** using currently set training data.

        On the other hand, caller can call ``fit`` multiple times on the same training data
        to continue fitting.

        If ``fit`` fully fits using provided training data, there is no point in making further
        calls to this method with same training data, and in fact further calls can be noops,
        or a primitive can decide to refit from scratch.

        In the case fitting can continue with same training data (even if it is maybe not reasonable,
        because the internal metric primitive is using looks like fitting will be degrading), if ``fit``
        is called again (without setting training data), the primitive has to continue fitting.

        Caller can provide ``timeout`` information to guide the length of the fitting process.
        Ideally, a primitive should adapt its fitting process to try to do the best fitting possible
        inside the time allocated. If this is not possible and the primitive reaches the timeout
        before fitting, it should raise a ``TimeoutError`` exception to signal that fitting was
        unsuccessful in the given time. The state of the primitive after the exception should be
        as the method call has never happened and primitive should continue to operate normally.
        The purpose of ``timeout`` is to give opportunity to a primitive to cleanly manage
        its state instead of interrupting execution from outside. Maintaining stable internal state
        should have precedence over respecting the ``timeout`` (caller can terminate the misbehaving
        primitive from outside anyway). If a longer ``timeout`` would produce different fitting,
        then ``CallResult``'s ``has_finished`` should be set to ``False``.

        Some primitives have internal fitting iterations (for example, epochs). For those, caller
        can provide how many of primitive's internal iterations should a primitive do before returning.
        Primitives should make iterations as small as reasonable. If ``iterations`` is ``None``,
        then there is no limit on how many iterations the primitive should do and primitive should
        choose the best amount of iterations on its own (potentially controlled through
        hyper-parameters). If ``iterations`` is a number, a primitive has to do those number of
        iterations (even if not reasonable), if possible. ``timeout`` should still be respected
        and potentially less iterations can be done because of that. Primitives with internal
        iterations should make ``CallResult`` contain correct values.

        For primitives which do not have internal iterations, any value of ``iterations``
        means that they should fit fully, respecting only ``timeout``.

        Parameters
        ----------
        timeout : float
            A maximum time this primitive should be fitting during this method call, in seconds.
        iterations : int
            How many of internal iterations should the primitive do.

        Returns
        -------
        CallResult[None]
            A ``CallResult`` with ``None`` value.
        """

    @abc.abstractmethod
    def get_params(self) -> Params:
        """
        Returns parameters of this primitive.

        Parameters are all parameters of the primitive which can potentially change during a life-time of
        a primitive. Parameters which cannot are passed through constructor.

        Parameters should include all data which is necessary to create a new instance of this primitive
        behaving exactly the same as this instance, when the new instance is created by passing the same
        parameters to the class constructor and calling ``set_params``.

        No other arguments to the method are allowed (except for private arguments).

        Returns
        -------
        Params
            An instance of parameters.
        """

    @abc.abstractmethod
    def set_params(self, *, params: Params) -> None:
        """
        Sets parameters of this primitive.

        Parameters are all parameters of the primitive which can potentially change during a life-time of
        a primitive. Parameters which cannot are passed through constructor.

        No other arguments to the method are allowed (except for private arguments).

        Parameters
        ----------
        params : Params
            An instance of parameters.
        """

    @classmethod
    def can_accept(cls, *, method_name: str, arguments: typing.Dict[str, typing.Union[metadata_module.Metadata, type]]) -> typing.Optional[metadata_module.DataMetadata]:
        """
        Returns a metadata object describing the output of a call of ``method_name`` method with
        arguments ``arguments``, if such arguments can be accepted by the method. Otherwise it
        returns ``None``.

        Default implementation checks structural types of ``arguments`` to match method's arguments' types.

        By (re)implementing this method, a primitive can fine-tune which arguments it accepts
        for its methods which goes beyond just structural type checking. For example, a primitive might
        operate only on images, so it can accept NumPy arrays, but only those with semantic type
        corresponding to an image. Or it might check dimensions of an array to assure it operates
        on square matrix.

        Parameters
        ----------
        method_name : str
            Name of the method which would be called.
        arguments : Dict[str, Union[Metadata, type]]
            A mapping between argument names and their metadata objects (for pipeline arguments) or types (for other).

        Returns
        -------
        DataMetadata
            Metadata object of the method call result, or ``None`` if arguments are not accepted
            by the method.
        """

        metadata = cls.metadata.query()

        primitive_arguments = metadata['primitive_code']['arguments']
        primitive_required_arguments_set = {argument_name for argument_name, argument in primitive_arguments.items() if 'default' not in argument}

        method = metadata['primitive_code']['instance_methods'].get(method_name, None)

        if method is None:
            return None

        method_arguments_set = set(method['arguments'])
        required_arguments_set = {method_name for method_name in method_arguments_set if method_name in primitive_required_arguments_set}
        arguments_keys_set = set(arguments.keys())

        # All required arguments should be present.
        if len(required_arguments_set - arguments_keys_set):
            return None

        # All passed arguments should exist among method arguments.
        if len(arguments_keys_set - method_arguments_set):
            return None

        for argument_name, argument_metadata in arguments.items():
            if primitive_arguments[argument_name]['kind'] == metadata_module.PrimitiveArgumentKind.PIPELINE:
                if not isinstance(argument_metadata, metadata_module.Metadata):
                    return None

                if isinstance(argument_metadata, metadata_module.PrimitiveMetadata):
                    argument_type = argument_metadata.query().get('structural_type', None)
                else:
                    argument_type = argument_metadata.query(()).get('structural_type', None)
            else:
                if not isinstance(argument_metadata, type):
                    return None

                argument_type = argument_metadata

            if argument_type is None:
                return None

            # TODO: Should we require here strict type equality?
            #       Strict type equality is easier and faster to check. And also authors can provide
            #       exact type they are using in their metadata annotation. Moreover, primitive
            #       validation checks where a type should be strict type equality and where we allow
            #       subtypes so we should not have to repeat this here again.
            if not utils.is_subclass(argument_type, primitive_arguments[argument_name]['type']):
                return None

        return metadata_module.DataMetadata({
            'schema': metadata_module.CONTAINER_SCHEMA_VERSION,
            'structural_type': method['returns'],
        })


class ContinueFitMixin(typing.Generic[Inputs, Outputs, Params, Hyperparams]):
    @abc.abstractmethod
    def continue_fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        """
        Similar to base ``fit``, this method fits the primitive using inputs and outputs (if any)
        using currently set training data.

        The difference is what happens when currently set training data is different from
        what the primitive might have already been fitted on. ``fit`` fits the primitive from
        scratch, while ``continue_fit`` fits it further and does **not** start from scratch.

        Caller can still call ``continue_fit`` multiple times on the same training data as well,
        in which case primitive should try to improve the fit in the same way as with ``fit``.

        From the perspective of a caller of all other methods, the training data in effect
        is still just currently set training data. If a caller wants to call ``gradient_output``
        on all data on which the primitive has been fitted through multiple calls of ``continue_fit``
        on different training data, the caller should pass all this data themselves through
        another call to ``set_training_data``, do not call ``fit`` or ``continue_fit`` again,
        and use ``gradient_output`` method. In this way primitives which truly support
        continuation of fitting and need only the latest data to do another fitting, do not
        have to keep all past training data around themselves.

        If a primitive supports this mixin, then both ``fit`` and ``continue_fit`` can be
        called. ``continue_fit`` always continues fitting, if it was started through ``fit``
        or ``continue_fit``. And ``fit`` always restarts fitting, even if previously
        ``continue_fit`` was used.

        Parameters
        ----------
        timeout : float
            A maximum time this primitive should be fitting during this method call, in seconds.
        iterations : int
            How many of internal iterations should the primitive do.

        Returns
        -------
        CallResult[None]
            A ``CallResult`` with ``None`` value.
        """


class SamplingCompositionalityMixin(typing.Generic[Inputs, Outputs, Params, Hyperparams]):
    """
    This mixin signals to a caller that the primitive is probabilistic but
    may be likelihood free.
    """

    @abc.abstractmethod
    def sample(self, *, inputs: Inputs, num_samples: int = 1, timeout: float = None, iterations: int = None) -> CallResult[typing.Sequence[Outputs]]:
        """
        Sample each input from ``inputs`` ``num_samples`` times.

        Semantics of ``timeout`` and ``iterations`` is the same as in ``produce``.

        Parameters
        ----------
        inputs : Inputs
            The inputs of shape [num_inputs, ...].
        num_samples : int
            The number of samples to return in a set of samples.
        timeout : float
            A maximum time this primitive should take to sample outputs during this method call, in seconds.
        iterations : int
            How many of internal iterations should the primitive do.

        Returns
        -------
        CallResult[Sequence[Outputs]]
            The multiple sets of samples of shape [num_samples, num_inputs, ...] wrapped inside
            ``CallResult``. While the output value type is specified as ``Sequence[Outputs]``, the
            output value can be in fact any container type with dimensions/shape equal to combined
            ``Sequence[Outputs]`` dimensions/shape. Subclasses should specify which exactly type
            the output is.
        """


class ProbabilisticCompositionalityMixin(typing.Generic[Inputs, Outputs, Params, Hyperparams]):
    """
    This mixin provides additional abstract methods which primitives should implement to
    help callers with doing various end-to-end refinements using probabilistic
    compositionality.

    This mixin adds methods to support at least:

    * Metropolis-Hastings

    Mixin should be used together with ``SamplingCompositionalityMixin`` mixin.
    """

    @abc.abstractmethod
    def log_likelihoods(self, *, outputs: Outputs, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[typing.Sequence[float]]:
        """
        Returns log probability of outputs given inputs and params under this primitive:

        log(p(output_i | input_i, params))

        Parameters
        ----------
        outputs : Outputs
            The outputs.
        inputs : Inputs
            The inputs.
        timeout : float
            A maximum time this primitive should take to produce outputs during this method call, in seconds.
        iterations : int
            How many of internal iterations should the primitive do.

        Returns
        -------
        CallResult[Sequence[float]]
            log(p(output_i | input_i, params))) wrapped inside ``CallResult``.
            While the output value type is specified as ``Sequence[float]``, the output
            value can be in fact any container type with dimensions/shape equal to
            combined ``Sequence[float]`` dimensions/shape. Subclasses should specify
            which exactly type the output is.
        """

    def log_likelihood(self, *, outputs: Outputs, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[float]:
        """
        Returns log probability of outputs given inputs and params under this primitive:

        sum_i(log(p(output_i | input_i, params)))

        By default it calls ``log_likelihoods`` and computes a sum, but subclasses can
        implement a more efficient version.

        Parameters
        ----------
        outputs : Outputs
            The outputs.
        inputs : Inputs
            The inputs.
        timeout : float
            A maximum time this primitive should take to produce outputs during this method call, in seconds.
        iterations : int
            How many of internal iterations should the primitive do.

        Returns
        -------
        CallResult[float]
            sum_i(log(p(output_i | input_i, params))) wrapped inside ``CallResult``.
        """

        result = self.log_likelihoods(outputs=outputs, inputs=inputs, timeout=timeout, iterations=iterations)

        return CallResult(sum(result.value), result.has_finished, result.iterations_done)


# TODO: This is not yet a properly defined type which would really be recognized as a version of Params.
#       You should specify a proper type in your subclass. Type checking might complain that your
#       type does not match the parent type, but ignore it (add "type: ignore" comment to that line).
#       This type will be fixed in the future.
class Scores(typing.Generic[Params]):
    """
    A type representing a version of ``Params`` which holds all the differentiable fields from
    ``Params`` but all their values are of type ``float``.
    """


Container = typing.TypeVar('Container', bound=types.Container)


# TODO: This is not yet a properly defined type which would really be recognized similar to Container.
#       You should specify a proper type in your subclass. Type checking might complain that your
#       type does not match the parent type, but ignore it (add "type: ignore" comment to that line).
#       This type will be fixed in the future.
class Gradients(typing.Generic[Container]):
    """
    A type representing a structure similar to ``Container``, but the values are of type ``Optional[float]``.
    Value is ``None`` if gradient for that part of the structure is not possible.
    """


class GradientCompositionalityMixin(typing.Generic[Inputs, Outputs, Params, Hyperparams]):
    """
    This mixin provides additional abstract methods which primitives should implement to
    help callers with doing various end-to-end refinements using gradient-based
    compositionality.

    This mixin adds methods to support at least:

    * gradient-based, compositional end-to-end training
    * regularized pre-training
    * multi-task adaptation
    * black box variational inference
    * Hamiltonian Monte Carlo
    """

    @abc.abstractmethod
    def gradient_output(self, *, outputs: Outputs, inputs: Inputs) -> Gradients[Outputs]:
        """
        Returns the gradient of loss sum_i(L(output_i, produce_one(input_i))) with respect to outputs.

        When fit term temperature is set to non-zero, it should return the gradient with respect to outputs of:

        sum_i(L(output_i, produce_one(input_i))) + temperature * sum_i(L(training_output_i, produce_one(training_input_i)))

        When used in combination with the ``ProbabilisticCompositionalityMixin``, it returns gradient
        of sum_i(log(p(output_i | input_i, params))) with respect to outputs.

        When fit term temperature is set to non-zero, it should return the gradient with respect to outputs of:

        sum_i(log(p(output_i | input_i, params))) + temperature * sum_i(log(p(training_output_i | training_input_i, params)))

        Parameters
        ----------
        outputs : Outputs
            The outputs.
        inputs : Inputs
            The inputs.

        Returns
        -------
        Gradients[Outputs]
            A structure similar to ``Container`` but the values are of type ``Optional[float]``.
        """

    @abc.abstractmethod
    def gradient_params(self, *, outputs: Outputs, inputs: Inputs) -> Scores[Params]:
        """
        Returns the gradient of loss sum_i(L(output_i, produce_one(input_i))) with respect to params.

        When fit term temperature is set to non-zero, it should return the gradient with respect to params of:

        sum_i(L(output_i, produce_one(input_i))) + temperature * sum_i(L(training_output_i, produce_one(training_input_i)))

        When used in combination with the ``ProbabilisticCompositionalityMixin``, it returns gradient of
        log(p(output | input, params)) with respect to params.

        When fit term temperature is set to non-zero, it should return the gradient with respect to params of:

        sum_i(log(p(output_i | input_i, params))) + temperature * sum_i(log(p(training_output_i | training_input_i, params)))

        Parameters
        ----------
        outputs : Outputs
            The outputs.
        inputs : Inputs
            The inputs.

        Returns
        -------
        Scores[Params]
            A version of ``Params`` with all differentiable fields from ``Params`` and values set to gradient for each parameter.
        """

    def forward(self, *, inputs: Inputs) -> Outputs:
        """
        Similar to ``produce`` method but it is meant to be used for a forward pass during
        backpropagation-based end-to-end training. Primitive can implement it differently
        than ``produce``, e.g., forward pass during training can enable dropout layers, or
        ``produce`` might not compute gradients while ``forward`` does.

        By default it calls ``produce`` for one iteration.

        Parameters
        ----------
        inputs : Inputs
            The inputs of shape [num_inputs, ...].

        Returns
        -------
        Outputs
            The outputs of shape [num_inputs, ...].
        """

        return self.produce(inputs=inputs, timeout=None, iterations=1).value  # type: ignore

    @abc.abstractmethod
    def backward(self, *, gradient_outputs: Gradients[Outputs], fine_tune: bool = False, fine_tune_learning_rate: float = 0.00001,
                 fine_tune_weight_decay: float = 0.00001) -> typing.Tuple[Gradients[Inputs], Scores[Params]]:
        """
        Returns the gradient with respect to inputs and with respect to params of a loss
        that is being backpropagated end-to-end in a pipeline.

        This is the standard backpropagation algorithm: backpropagation needs to be preceded by a
        forward propagation (``forward`` method).

        Parameters
        ----------
        gradient_outputs : Gradients[Outputs]
            The gradient of the loss with respect to this primitive's output. During backpropagation,
            this comes from the next primitive in the pipeline, i.e., the primitive whose input
            is the output of this primitive during the forward execution with ``forward`` (and ``produce``).
        fine_tune : bool
            If ``True``, executes a fine-tuning gradient descent step as a part of this call.
            This provides the most straightforward way of end-to-end training/fine-tuning.
        fine_tune_learning_rate : float
            Learning rate for end-to-end training/fine-tuning gradient descent steps.
        fine_tune_weight_decay : float
            L2 regularization (weight decay) coefficient for end-to-end training/fine-tuning gradient
            descent steps.

        Returns
        -------
        Tuple[Gradients[Inputs], Scores[Params]]
            A tuple of the gradient with respect to inputs and with respect to params.
        """

    @abc.abstractmethod
    def set_fit_term_temperature(self, *, temperature: float = 0) -> None:
        """
        Sets the temperature used in ``gradient_output`` and ``gradient_params``.

        Parameters
        ----------
        temperature : float
            The temperature to use, [0, inf), typically, [0, 1].
        """


class LossFunctionMixin(typing.Generic[Inputs, Outputs, Params, Hyperparams]):
    """
    Mixin which provides abstract methods for a caller to call to inspect which
    loss function a primitive is using internally, and to compute loss on given
    inputs and outputs.
    """

    @abc.abstractmethod
    def get_loss_function(self) -> typing.Optional[problem.PerformanceMetric]:
        """
        Returns a D3M metric value of the loss function used by the primitive during the last fitting,
        or ``None`` if using a non-standard loss function or if the primitive does not use a loss
        function at all.

        Returns
        -------
        Metric
            A D3M standard metric value of the loss function used.
        """

    @abc.abstractmethod
    def get_loss_primitive(self) -> typing.Optional[PrimitiveBase]:
        """
        Primitives can be passed to other primitives as arguments. As such, some primitives
        can accept another primitive as a loss function to use, or use it internally. This
        method allows a primitive to expose this loss primitive to others, returning directly
        an instance of the primitive being used during the last fitting.

        Returns
        -------
        PrimitiveBase
            A D3M primitive used to compute loss.
        """

    @abc.abstractmethod
    def losses(self, *, inputs: Inputs, outputs: Outputs, timeout: float = None, iterations: int = None) -> CallResult[typing.Sequence[float]]:
        """
        Returns the loss L(output_i, produce_one(input_i)) for each (input_i, output_i) pair
        using a loss function used by the primitive during the last fitting.

        Parameters
        ----------
        inputs : Inputs
            The inputs.
        outputs : Outputs
            The outputs.
        timeout : float
            A maximum time this primitive should take to produce outputs during this method call, in seconds.
        iterations : int
            How many of internal iterations should the primitive do.

        Returns
        -------
        CallResult[Sequence[float]]
            L(output_i, produce_one(input_i)) for each (input_i, output_i) pair
            wrapped inside ``CallResult``.
            While the output value type is specified as ``Sequence[float]``, the output
            value can be in fact any container type with dimensions/shape equal to
            combined ``Sequence[float]`` dimensions/shape. Subclasses should specify
            which exactly type the output is.
        """

    def loss(self, *, inputs: Inputs, outputs: Outputs, timeout: float = None, iterations: int = None) -> CallResult[float]:
        """
        Returns the loss sum_i(L(output_i, produce_one(input_i))) for all (input_i, output_i) pairs
        using a loss function used by the primitive during the last fitting.

        By default it calls ``losses`` and computes a sum, but subclasses can implement
        a more efficient version.

        Parameters
        ----------
        inputs : Inputs
            The inputs.
        outputs : Outputs
            The outputs.
        timeout : float
            A maximum time this primitive should take to produce outputs during this method call, in seconds.
        iterations : int
            How many of internal iterations should the primitive do.

        Returns
        -------
        CallResult[float]
            sum_i(L(output_i, produce_one(input_i))) for all (input_i, output_i) pairs
            wrapped inside ``CallResult``.
        """

        result = self.losses(inputs=inputs, outputs=outputs, timeout=timeout, iterations=iterations)

        return CallResult(sum(result.value), result.has_finished, result.iterations_done)


class SingletonOutputMixin(typing.Generic[Inputs, Outputs, Params, Hyperparams]):
    """
    If a primitive is inheriting from this mixin, it is signaling that all outputs from
    its ``produce`` method (and other extra "produce" methods) are sequences of
    length 1. This is useful because a caller can then directly extract this element.

    Example of such primitives are primitives which compute loss, which are returning one number
    for multiple inputs. With this mixin they can return a sequence with this one number, but
    caller which cares about the loss can extract it out. At the same time, other callers which
    operate only on sequences can continue to operate normally.

    We can see other primitives as mapping primitives, and primitives with this mixin as
    reducing primitives.
    """
