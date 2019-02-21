## vNEXT

* Added `RandomPrimitive` test primitive.

## v2018.1.5

* Made use of the PyPI package official. Documented a requirement for
  `--process-dependency-links` argument during installation.
* Arguments `learning_rate` and `weight_decay` in `GradientCompositionalityMixin` renamed to
  `fine_tune_learning_rate` and `fine_tune_weight_decay`, respectively.
  `learning_rate` is a common hyper-parameter name.
  [#41](https://gitlab.com/datadrivendiscovery/primitive-interfaces/issues/41)

## v2017.12.27

* Documented `produce` method for `ClusteringPrimitiveBase` and added
  `ClusteringDistanceMatrixMixin`.
  [#18](https://gitlab.com/datadrivendiscovery/primitive-interfaces/issues/18)
* Added `can_accept` class method to primitive base class and implemented its
  default implementation.
  [#20](https://gitlab.com/datadrivendiscovery/primitive-interfaces/issues/20)
* "Distance" primitives now accept an extra argument instead of a tuple.
* `Params` should now be a subclass of `d3m_metadata.params.Params`, which is a
  specialized dict instead of a named tuple.
* Removed `Graph` class. There is no need for it anymore because we can identify
  them by having input type a NetworkX graph and through metadata discovery.
* Added `timeout` and `iterations` arguments to more methods.
* Added `forward` and `backward` backprop methods to `GradientCompositionalityMixin`
  to allow end-to-end backpropagation across diverse primitives.
  [#26](https://gitlab.com/datadrivendiscovery/primitive-interfaces/issues/26)
* Added `log_likelihoods` method to `ProbabilisticCompositionalityMixin`.
* Constructor now accepts `docker_containers` argument with addresses of running
  primitive's Docker containers.
  [#25](https://gitlab.com/datadrivendiscovery/primitive-interfaces/issues/25)
* Removed `CallMetadata` and `get_call_metadata` and changed so that some methods
  directly return new but similar `CallResult`.
  [#27](https://gitlab.com/datadrivendiscovery/primitive-interfaces/issues/27)
* Documented how extra arguments to standard and extra methods can be defined.
* Documented that all arguments with the same name in all methods should have the
  same type. Arguments are per primitive not per method.
  [#29](https://gitlab.com/datadrivendiscovery/primitive-interfaces/issues/29)
* Specified how to define extra "produce" methods which have same semantics
  as `produce` but different output types.
  [#30](https://gitlab.com/datadrivendiscovery/primitive-interfaces/issues/30)
* Added `SingletonOutputMixin` to signal that primitive's output contains
  only one element.
  [#15](https://gitlab.com/datadrivendiscovery/primitive-interfaces/issues/15)  
* Added `get_loss_primitive` to allow accessing to the loss primitive
  being used.
* Moved `set_training_data` back to the base class.
  This breaks Liskov substitution principle.
  [#19](https://gitlab.com/datadrivendiscovery/primitive-interfaces/issues/19)
* Renamed `__metadata__` to `metadata` attribute.
  [#23](https://gitlab.com/datadrivendiscovery/primitive-interfaces/issues/23)
* Package now requires Python 3.6.
* Repository migrated to gitlab.com and made public.
* `set_random_seed` method has been removed and replaced with a
  `random_seed` argument to the constructor, which is also exposed as an attribute.
  [#16](https://gitlab.com/datadrivendiscovery/primitive-interfaces/issues/16)
* Primitives have now `hyperparams` attribute which returns a
  hyper-parameters object passed to the constructor.
  [#14](https://gitlab.com/datadrivendiscovery/primitive-interfaces/issues/14)
* `Params` and `Hyperparams` are now required to be pickable and copyable.
  [#3](https://gitlab.com/datadrivendiscovery/primitive-interfaces/issues/3)
* Primitives are now parametrized by `Hyperparams` type variable as well.
  Constructor now receives hyper-parameters as an instance as one argument
  instead of multiple keyword arguments.
  [#13](https://gitlab.com/datadrivendiscovery/primitive-interfaces/issues/13)
* `LossFunctionMixin`'s `get_loss_function` method now returns a value from
  problem schema `Metric` enumeration.
* `LossFunctionMixin` has now a `loss` and `losses` methods which allows one
  to ask a primitive to compute loss for a given set of inputs and outputs using
  internal loss function the primitive is using.
  [#17](https://gitlab.com/datadrivendiscovery/primitive-interfaces/issues/17)

## v2017.9.27

* Initial version of the unified Python API.
