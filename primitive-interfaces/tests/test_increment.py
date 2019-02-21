import json
import unittest
import os.path
import sys

import numpy

import primitive_interfaces
from d3m_metadata import container, metadata, utils

TEST_PRIMITIVES_DIR = os.path.join(os.path.dirname(__file__), 'data', 'primitives')

sys.path.insert(0, TEST_PRIMITIVES_DIR)

from test_primitives.increment import IncrementPrimitive


EXPECTED_PRIMITIVE_DESCRIPTION_JSON = r"""
{
    "id": "5c9d5acf-7754-420f-a49f-90f4d9d0d694",
    "version": "0.1.0",
    "name": "Increment Values",
    "keywords": [
        "test primitive"
    ],
    "source": {
        "name": "Test team",
        "uris": [
            "https://gitlab.com/datadrivendiscovery/tests-data/blob/master/primitives/test_primitives/increment.py",
            "https://gitlab.com/datadrivendiscovery/tests-data.git"
        ]
    },
    "installation": [
        {
            "type": "PIP",
            "package_uri": "git+https://gitlab.com/datadrivendiscovery/tests-data.git@__GIT_COMMIT__#egg=test_primitives&subdirectory=primitives"
        }
    ],
    "location_uris": [
        "https://gitlab.com/datadrivendiscovery/tests-data/raw/__GIT_COMMIT__/primitives/test_primitives/increment.py"
    ],
    "python_path": "d3m.primitives.test.IncrementPrimitive",
    "algorithm_types": [
        "COMPUTER_ALGEBRA"
    ],
    "primitive_family": "OPERATOR",
    "preconditions": [
        "NO_MISSING_VALUES",
        "NO_CATEGORICAL_VALUES"
    ],
    "schema": "https://metadata.datadrivendiscovery.org/schemas/v0/primitive.json",
    "original_python_path": "test_primitives.increment.IncrementPrimitive",
    "primitive_code": {
        "class_type_arguments": {
            "Inputs": "d3m_metadata.container.numpy.ndarray",
            "Outputs": "d3m_metadata.container.numpy.ndarray",
            "Hyperparams": "test_primitives.increment.Hyperparams",
            "Params": "NoneType"
        },
        "interfaces_version": "__INTERFACES_VERSION__",
        "interfaces": [
            "transformer.TransformerPrimitiveBase",
            "base.PrimitiveBase"
        ],
        "hyperparams": {
            "amount": {
                "type": "d3m_metadata.hyperparams.Hyperparameter",
                "default": 1,
                "structural_type": "float",
                "semantic_types": [
                    "https://metadata.datadrivendiscovery.org/types/ControlParameter"
                ]
            }
        },
        "arguments": {
            "hyperparams": {
                "type": "test_primitives.increment.Hyperparams",
                "kind": "RUNTIME"
            },
            "random_seed": {
                "type": "int",
                "kind": "RUNTIME",
                "default": 0
            },
            "docker_containers": {
                "type": "typing.Union[typing.Dict[str, str], NoneType]",
                "kind": "RUNTIME",
                "default": "None"
            },
            "timeout": {
                "type": "typing.Union[float, NoneType]",
                "kind": "RUNTIME",
                "default": "None"
            },
            "iterations": {
                "type": "typing.Union[int, NoneType]",
                "kind": "RUNTIME",
                "default": "None"
            },
            "inputs": {
                "type": "d3m_metadata.container.numpy.ndarray",
                "kind": "PIPELINE"
            },
            "params": {
                "type": "NoneType",
                "kind": "RUNTIME"
            }
        },
        "class_methods": {
            "can_accept": {
                "arguments": {
                    "method_name": {
                        "type": "str"
                    },
                    "arguments": {
                        "type": "typing.Dict[str, typing.Union[d3m_metadata.metadata.Metadata, type]]"
                    }
                },
                "returns": "typing.Union[d3m_metadata.metadata.DataMetadata, NoneType]"
            }
        },
        "instance_methods": {
            "__init__": {
                "kind": "OTHER",
                "arguments": [
                    "hyperparams",
                    "random_seed",
                    "docker_containers"
                ],
                "returns": "NoneType",
                "description": "All primitives should accept all their hyper-parameters in a constructor as one value,\nan instance of type ``Hyperparams``.\n\nMethods can accept per-call overrides for some hyper-parameters (e.g., a threshold\nfor fitting). Those arguments have to match in name and type a hyper-parameter defined\nin ``hyperparams`` object. The value provided in the ``hyperparams`` object serves\nas a default in such case.\n\nProvided random seed should control all randomness used by this primitive.\nPrimitive should behave exactly the same for the same random seed across multiple\ninvocations. You can call `numpy.random.RandomState(random_seed)` to obtain an\ninstance of a random generator using provided seed.\n\nPrimitives can be wrappers around or use one or more Docker images which they can\nspecify as part of  ``installation`` field in their metadata. Each Docker image listed\nthere has a ``key`` field identifying that image. When primitive is created,\n``docker_containers`` contains a mapping between those keys and addresses to which\nprimitive can connect to access a running Docker container for a particular Docker\nimage. Docker containers might be long running and shared between multiple instances\nof a primitive.\n\nNo other arguments to the constructor are allowed (except for private arguments)\nbecause we want instances of primitives to be created without a need for any other\nprior computation.\n\nConstructor should be kept lightweight and not do any computation. No resources\nshould be allocated in the constructor. Resources should be allocated when needed."
            },
            "fit": {
                "kind": "OTHER",
                "arguments": [
                    "timeout",
                    "iterations"
                ],
                "returns": "NoneType",
                "description": "A noop."
            },
            "get_params": {
                "kind": "OTHER",
                "arguments": [],
                "returns": "NoneType",
                "description": "A noop."
            },
            "produce": {
                "kind": "PRODUCE",
                "arguments": [
                    "inputs",
                    "timeout",
                    "iterations"
                ],
                "returns": "primitive_interfaces.base.CallResult[d3m_metadata.container.numpy.ndarray]"
            },
            "set_params": {
                "kind": "OTHER",
                "arguments": [
                    "params"
                ],
                "returns": "NoneType",
                "description": "A noop."
            },
            "set_training_data": {
                "kind": "OTHER",
                "arguments": [],
                "returns": "NoneType",
                "description": "A noop."
            }
        },
        "class_attributes": {
            "metadata": "d3m_metadata.metadata.PrimitiveMetadata"
        },
        "instance_attributes": {
            "hyperparams": "d3m_metadata.hyperparams.Hyperparams",
            "random_seed": "int",
            "docker_containers": "typing.Dict[str, str]"
        }
    },
    "structural_type": "test_primitives.increment.IncrementPrimitive",
    "description": "A primitive which increments each value by a fixed amount, by default 1."
}
""".replace('__INTERFACES_VERSION__', primitive_interfaces.__version__).replace('__GIT_COMMIT__', utils.current_git_commit(TEST_PRIMITIVES_DIR))


class TestIncrementPrimitive(unittest.TestCase):
    # It is not necessary to call "can_accept" before calling a method
    # during runtime, but we are doing it here for testing purposes.
    def call_primitive(self, primitive, method_name, **kwargs):
        primitive_arguments = primitive.metadata.query()['primitive_code']['arguments']

        arguments = {}

        for argument_name, argument_value in kwargs.items():
            if primitive_arguments[argument_name]['kind'] == metadata.PrimitiveArgumentKind.PIPELINE:
                arguments[argument_name] = argument_value.metadata
            else:
                arguments[argument_name] = type(argument_value)

        self.assertTrue(type(primitive).can_accept(
            method_name=method_name,
            arguments=arguments,
        ))

        return getattr(primitive, method_name)(**kwargs)

    def test_basic(self):
        hyperparams_class = IncrementPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']

        primitive = IncrementPrimitive(hyperparams=hyperparams_class.defaults())

        inputs = container.ndarray([[1, 2, 3, 4], [5, 6, 7, 8]], {
            'schema': metadata.CONTAINER_SCHEMA_VERSION,
            'structural_type': container.ndarray,
            'dimension': {
                'length': 2,
            },
        })
        inputs.metadata = inputs.metadata.update((metadata.ALL_ELEMENTS,), {
            'dimension': {
                'length': 4,
            },
        })
        inputs.metadata = inputs.metadata.update((metadata.ALL_ELEMENTS, metadata.ALL_ELEMENTS), {
            'structural_type': inputs.dtype.type,
        })

        call_metadata = self.call_primitive(primitive, 'produce', inputs=inputs)

        self.assertTrue(numpy.array_equal(call_metadata.value, container.ndarray([[2, 3, 4, 5], [6, 7, 8, 9]])))
        self.assertEqual(call_metadata.has_finished, True)
        self.assertEqual(call_metadata.iterations_done, None)

        self.assertIs(call_metadata.value.metadata.for_value, call_metadata.value)
        self.assertEqual(call_metadata.value.metadata.query(())['dimension']['length'], 2)
        self.assertEqual(call_metadata.value.metadata.query((metadata.ALL_ELEMENTS,))['dimension']['length'], 4)
        self.assertEqual(call_metadata.value.metadata.query((metadata.ALL_ELEMENTS, metadata.ALL_ELEMENTS))['structural_type'], numpy.int64)

    def test_hyperparameter(self):
        hyperparams_class = IncrementPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']

        primitive = IncrementPrimitive(hyperparams=hyperparams_class(amount=2))

        inputs = container.ndarray([[1, 2, 3, 4], [5, 6, 7, 8]], {
            'schema': metadata.CONTAINER_SCHEMA_VERSION,
            'structural_type': container.ndarray,
            'dimension': {
                'length': 2,
            },
        })
        inputs.metadata = inputs.metadata.update((metadata.ALL_ELEMENTS,), {
            'dimension': {
                'length': 4,
            },
        })
        inputs.metadata = inputs.metadata.update((metadata.ALL_ELEMENTS, metadata.ALL_ELEMENTS), {
            'structural_type': inputs.dtype.type,
        })

        call_metadata = self.call_primitive(primitive, 'produce', inputs=inputs)

        self.assertTrue(numpy.array_equal(call_metadata.value, container.ndarray([[3, 4, 5, 6], [7, 8, 9, 10]])))
        self.assertEqual(call_metadata.has_finished, True)
        self.assertEqual(call_metadata.iterations_done, None)

        self.assertIs(call_metadata.value.metadata.for_value, call_metadata.value)
        self.assertEqual(call_metadata.value.metadata.query(())['dimension']['length'], 2)
        self.assertEqual(call_metadata.value.metadata.query((metadata.ALL_ELEMENTS,))['dimension']['length'], 4)
        self.assertEqual(call_metadata.value.metadata.query((metadata.ALL_ELEMENTS, metadata.ALL_ELEMENTS))['structural_type'], numpy.int64)

    def test_can_accept(self):
        inputs_metadata = metadata.DataMetadata({
            'schema': metadata.CONTAINER_SCHEMA_VERSION,
            'structural_type': container.ndarray,
        })

        self.assertFalse(IncrementPrimitive.can_accept(method_name='produce', arguments={
            'inputs': inputs_metadata,
        }))

        inputs_metadata.update((), {
            'dimension': {
                'length': 2,
            },
        })

        self.assertFalse(IncrementPrimitive.can_accept(method_name='produce', arguments={
            'inputs': inputs_metadata,
        }))

        inputs_metadata.update((metadata.ALL_ELEMENTS,), {
            'dimension': {
                'length': 2,
            },
        })

        self.assertFalse(IncrementPrimitive.can_accept(method_name='produce', arguments={
            'inputs': inputs_metadata,
        }))

        inputs_metadata.update((metadata.ALL_ELEMENTS, metadata.ALL_ELEMENTS), {
            'structural_type': str,
        })

        self.assertFalse(IncrementPrimitive.can_accept(method_name='produce', arguments={
            'inputs': inputs_metadata,
        }))

        inputs_metadata.update((metadata.ALL_ELEMENTS, metadata.ALL_ELEMENTS), {
            'structural_type': float,
        })

        self.assertFalse(IncrementPrimitive.can_accept(method_name='produce', arguments={
            'inputs': inputs_metadata,
        }))

    def test_metadata(self):
        expected_description = json.loads(EXPECTED_PRIMITIVE_DESCRIPTION_JSON)

        # We stringify to JSON and parse it to make sure the description can be stringified to JSON.
        description = json.loads(json.dumps(IncrementPrimitive.metadata.to_json()))

        self.maxDiff = None
        self.assertEqual(expected_description, description)


if __name__ == '__main__':
    unittest.main()
