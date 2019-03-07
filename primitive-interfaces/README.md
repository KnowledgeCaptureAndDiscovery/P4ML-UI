# Python interfaces for TA1 primitives

A collection of standard Python interfaces for TA1 primitives. All primitives should
extend one of the base classes available and optionally implement available mixins.

## About Data Driven Discovery Program

DARPA Data Driven Discovery (D3M) Program is researching ways to get machines to build
machine learning pipelines automatically. It is split into three layers:
TA1 (primitives), TA2 (systems which combine primitives automatically into pipelines
and executes them), and TA3 (end-users interfaces).

## Installation

This package works with Python 3.6+.

You can install latest stable version from [PyPI](https://pypi.python.org/pypi):

```
$ pip install --process-dependency-links primitive_interfaces
```

To install latest development version:

```
$ pip install --process-dependency-links git+https://gitlab.com/datadrivendiscovery/primitive-interfaces.git@devel
```

`--process-dependency-links` argument is required for correct processing of dependencies.

## Changelog

See [HISTORY.md](./HISTORY.md) for summary of changes to this package.

## Repository structure

`master` branch contains latest stable release of the package.
`devel` branch is a staging branch for the next release.

Releases are [tagged](https://gitlab.com/datadrivendiscovery/primitive-interfaces/tags).

## Design principles

Standard TA1 primitive interfaces have been designed to be possible for TA2 systems to call
primitives automatically and combine them into pipelines.

Some design principles applied:

* Use of a de facto standard language for "glue" between different components and libraries,
  Python.
* Use of keyword-only arguments for all methods so that caller does not have to worry
  about the order of arguments.
* Every primitive should implement only one functionality, more or less a function, with clear
  inputs and outputs. All parameters of the function do not have to be known in advance and function
  can be "fitted" as part of the training step of the pipeline.
* Use of Python 3 typing extensions to annotate methods and classes with typing information
  to make it easier for TA2 systems to prune incompatible combinations of inputs and outputs
  and to reuse existing Python type-checking tooling.
* Typing information can serve both detecting issues and incompatibilities in primitive
  implementations and help with pipeline construction.
* All values being passed through a primitive have metadata associated with them.
* Primitives can operate only at a metadata level to help guide the pipeline construction
  process without having to operate on data itself.
* Primitive metadata is close to the source, primitive code, and not in separate files to minimize
  chances that it is goes out of sync. Metadata which can be automatically determined from the
  code should be automatically determined from the code. Similarly for data metadata.
* All randomness of primitives is captured by a random seed argument to assure reproducibility.
* Operations can work in iterations, under time budgets, and caller might not always want to
  compute values fully.
* Through use of mixins primitives can signal which capabilities they support.
* Primitives are to be composed and executed in a data-flow manner.

## Main concepts

Interface classes, mixins, and methods are documented in detail through use of docstrings and
typing annotations. Here we note some higher-level concept which can help understand basic
ideas behind interfaces and what they are trying to achieve, the big picture. This section
is not normative.

A primitive should extend one of the base classes available and optionally mixins as well.
Not all mixins apply to all primitives. That being said, you probably do not want to
subclass `PrimitiveBase` directly, but instead one of other base classes to signal to
a caller more about what your primitive is doing. If your primitive belong to a larger
set of primitives no exiting non-`PrimitiveBase` base class suits well, consider suggesting
that a new base class is created by opening an issue or making a merge request.

Base class and mixins have generally four type arguments you have to provide: `Inputs`,
`Outpus`, `Params`, and `Hyperparms`. One can see a primitive as parameterized by those
four type arguments. You can access them at runtime through metadata:

```python
FooBarPrimitive.metadata.query()['class_type_arguments']
```

`Inputs` should be set to a primary input type of a primitive. Primary, because you can
define additional inputs your primitive might need, but we will go into these details later.
Similarly for `Outputs`. `produce` method then produces outputs from inputs. Other primitive
methods help the primitive (and its `produce` method) achieve that, or help the runtime execute
the primitive as a whole, or optimize its behavior.

Both `Inputs` and `Outputs` should be of a
[standard container data type](https://gitlab.com/datadrivendiscovery/metadata#container-types).
We allow a limited set of value types being passed between primitives so that both TA2 and TA3
systems can implement introspection for those values if needed, or user interface for them, etc.
Moreover this allows us also to assure that they can be efficiently used with
Arrow/Plasma store.

Container values can then in turn contain values of
an [extended but still limited set of data types](https://gitlab.com/datadrivendiscovery/metadata#data-types).

Those values being passed between primitives also hold metadata. Metadata is available on
their `metadata` attribute. Metadata on values is stored in an instance of
[`DataMetadata`](https://gitlab.com/datadrivendiscovery/metadata/blob/master/d3m_metadata/metadata.py)
class. This is a reason why we have
[our own versions of some standard container types](https://gitlab.com/datadrivendiscovery/metadata/tree/master/d3m_metadata/container):
to have the `metadata` attribute.

All metadata is immutable and updating a metadata object returns a new, updated, copy.
Metadata internally remembers the history of changes, but there is no API yet to access that.
But the idea is that you will be able to follow the whole history of change to data in a pipeline
through metadata. See [metadata API](https://gitlab.com/datadrivendiscovery/metadata#api) for
more information how to manipulate metadata.

Primitives have a similar class `PrimitiveMetadata`, which when created automatically analyses
its primitive and populates parts of metadata based on that. In this way author does not have
to have information in two places (metadata and code) but just in code and metadata is extracted
from it. When possible. Some metadata author of the primitive stil has to provide directly.

Currently most standard interface base classes have only one `produce` method, but design
allows for multiple: their name has to be prefixed with `produce_`, have similar arguments
and same semantics as all produce methods. The main motivation for this is that some primitives
might be able to expose same results in different ways. Having multiple produce methods
allow the caller to pick which type of the result they want.

To keep primitive from outside simple and allow easier compositionality in pipelines, primitives
have arguments defined per primitive and not per their method. The idea here is that once a
caller satisfies (computes a value to be passed to) an argument, any method which requires
that argument can be called on a primitive.

There are three types of arguments:

* pipeline – arguments which are provided by the pipeline, they are required (otherwise caller would be able to trivially satisfy them by
  always passing `None` or another default value)
* runtime – arguments which caller provides during pipeline execution and they control
  various aspects of the execution
* hyper-parameter – a method can declare that primitive's hyper-parameter can be overridden for
  the call of the method, they have to match hyper-parameter definition

Methods can accept additional pipeline and hyper-parameter arguments and not just those from the
standard interfaces.

Because methods can accept additional arguments and because structural types are not enough
in many cases to know if a value is a good for a particular argument, there is a
`can_accept` class method which primitives can override to give the caller feedback in advance,
before the pipeline itself is already running. Default implementation jut checks structural
typing information and passes outputs structural typing information on, but ideally this
class method should check much more: shapes, dimensions, types of internal data structures, etc.
For example, the fact that inputs are a numpy array does not help much to know which value
to bring as inputs, because numpy array structural type does not cary enough information.
But metadata associated with values hopefully does.

Produce methods and some other methods return results wrapped in `CallResult`. In this way
primitives can expose information about internal iterative or optimization process and
allow caller to decide how long to run.

When calling a primitive, to access `Hyperparams` class you can do:

```python
hyperparams_class = FooBarPrimitive.metadata.query()['class_type_arguments']['Hyperparms']
```

You can now create an instance of the class by directly providing values for hyper-parameters,
use available simple sampling, or just use default values:

```python
hp1 = hyperparams_class({'threshold': 0.01})
hp2 = hyperparams_class.sample(random_state=42)
hp3 = hyperparams_class.defaults
```

You can then pass those instances as the `hyperparams` argument to primitive's constructor.

Author of a primitive has to define what internal parameters does the primitive have, if
any, by extending the `Params` class. It is just a fancy dict, so you can both create
an instance of it in the same way, and access its values:

```python
class Params(params.Params):
    coefficients: numpy.ndarray

ps = Params({'coefficients': numpy.array[1, 2, 3]})
ps['coefficients']
```

`Hyperparams` class and `Params` class have to be pickable and copyable so that instances
of primitives can be serialized and restored as needed.

Primitives (and some other values) are uniquely identified by their ID and version.
ID does not change through versions.

Primitives should not modify in-place any input argument but always first make a copy
before any modification.

## Examples

Examples of simple primitives using these interfaces can be found
[in this repository](https://gitlab.com/datadrivendiscovery/tests-data/tree/master/primitives):

* [MonomialPrimitive](https://gitlab.com/datadrivendiscovery/tests-data/blob/master/primitives/test_primitives/monomial.py)
  is a simple regressor which shows how to use `container.List`, define and use `Params`
  and `Hyperparams`, and implement multiple methods needed by a supervised learner primitive
* [IncrementPrimitive](https://gitlab.com/datadrivendiscovery/tests-data/blob/master/primitives/test_primitives/increment.py)
  is a transformer and shows how to have `container.ndarray` as inputs and outputs, how to set metadata for outputs,
  and how to extend `can_accept` to check if inputs have the right structure and types
  because `container.ndarray` does not expose those details through its structural type,
  but inputs' metadata provides it
* [SumPrimitive](https://gitlab.com/datadrivendiscovery/tests-data/blob/master/primitives/test_primitives/sum.py)
  is a transformer as well, but it is just a wrapper around a Docker image, it shows how
  to define Docker image in metadata and how to connect to a running Docker container,
  moreover, it also shows how inputs can be a union type of multiple other types
* [RandomPrimitive](https://gitlab.com/datadrivendiscovery/tests-data/blob/master/primitives/test_primitives/random.py)
  is a generator which shows how to use `random_seed`, too.
