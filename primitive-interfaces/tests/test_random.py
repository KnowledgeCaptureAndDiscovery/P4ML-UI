import json
import unittest
import os.path
import sys

import numpy

import primitive_interfaces
from d3m_metadata import container, metadata, utils

TEST_PRIMITIVES_DIR = os.path.join(os.path.dirname(__file__), 'data', 'primitives')

sys.path.insert(0, TEST_PRIMITIVES_DIR)

from test_primitives.random import RandomPrimitive


EXPECTED_PRIMITIVE_DESCRIPTION_JSON = r"""
{
    "id": "df3153a1-4411-47e2-bbc0-9d5e9925ad79",
    "version": "0.1.0",
    "name": "Random Samples",
    "keywords": [
        "test primitive"
    ],
    "source": {
        "name": "Test team",
        "uris": [
            "https://gitlab.com/datadrivendiscovery/tests-data/blob/master/primitives/test_primitives/random.py",
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
        "https://gitlab.com/datadrivendiscovery/tests-data/raw/__GIT_COMMIT__/primitives/test_primitives/random.py"
    ],
    "python_path": "d3m.primitives.test.RandomPrimitive",
    "algorithm_types": [
        "MERSENNE_TWISTER"
    ],
    "primitive_family": "DATA_GENERATION",
    "schema": "https://metadata.datadrivendiscovery.org/schemas/v0/primitive.json",
    "original_python_path": "test_primitives.random.RandomPrimitive",
    "primitive_code": {
        "class_type_arguments": {
            "Inputs": "d3m_metadata.container.list.List[NoneType]",
            "Outputs": "d3m_metadata.container.numpy.ndarray",
            "Hyperparams": "test_primitives.random.Hyperparams",
            "Params": "NoneType"
        },
        "interfaces_version": "__INTERFACES_VERSION__",
        "interfaces": [
            "generator.GeneratorPrimitiveBase",
            "base.PrimitiveBase"
        ],
        "hyperparams": {
            "mu": {
                "type": "d3m_metadata.hyperparams.Hyperparameter",
                "default": 0.0,
                "structural_type": "float",
                "semantic_types": [
                    "https://metadata.datadrivendiscovery.org/types/ControlParameter",
                    "https://metadata.datadrivendiscovery.org/types/TuningParameter"
                ]            
            },
            "sigma": {
                "type": "d3m_metadata.hyperparams.Hyperparameter",
                "default": 1.0,
                "structural_type": "float",
                "semantic_types": [
                    "https://metadata.datadrivendiscovery.org/types/ControlParameter",
                    "https://metadata.datadrivendiscovery.org/types/TuningParameter"
                ]            
            }
        },
        "arguments": {
            "hyperparams": {
                "type": "test_primitives.random.Hyperparams",
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
                "type": "d3m_metadata.container.list.List[NoneType]",
                "kind": "PIPELINE"
            },
            "params": {
                "type": "NoneType",
                "kind": "RUNTIME"
            }
        },
        "class_methods": {
            "can_accept": {
                "description": "Returns a metadata object describing the output of a call of ``method_name`` method with\narguments ``arguments``, if such arguments can be accepted by the method. Otherwise it\nreturns ``None``.\n\nDefault implementation checks structural types of ``arguments`` to match method's arguments' types.\n\nBy (re)implementing this method, a primitive can fine-tune which arguments it accepts\nfor its methods which goes beyond just structural type checking. For example, a primitive might\noperate only on images, so it can accept NumPy arrays, but only those with semantic type\ncorresponding to an image. Or it might check dimensions of an array to assure it operates\non square matrix.\n\nParameters\n----------\nmethod_name : str\n    Name of the method which would be called.\narguments : Dict[str, Union[Metadata, type]]\n    A mapping between argument names and their metadata objects (for pipeline arguments) or types (for other).\n\nReturns\n-------\nDataMetadata\n    Metadata object of the method call result, or ``None`` if arguments are not accepted\n    by the method.",
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
                "returns": "NoneType"
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
    "structural_type": "test_primitives.random.RandomPrimitive",
    "description": "A primitive which draws random samples from a normal distribution."
}
""".replace('__INTERFACES_VERSION__', primitive_interfaces.__version__).replace('__GIT_COMMIT__', utils.current_git_commit(TEST_PRIMITIVES_DIR))


class TestRandomPrimitive(unittest.TestCase):
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
        hyperparams_class = RandomPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']

        primitive = RandomPrimitive(random_seed=42, hyperparams=hyperparams_class.defaults())

        inputs = container.List[None]([None, None, None, None], {
            'schema': metadata.CONTAINER_SCHEMA_VERSION,
            'structural_type': container.List[None],
            'dimension': {
                'length': 4,
            },
        })
        inputs.metadata = inputs.metadata.update((metadata.ALL_ELEMENTS,), {
            'structural_type': type(None),
        })

        call_metadata = self.call_primitive(primitive, 'produce', inputs=inputs)

        self.assertTrue(numpy.allclose(call_metadata.value, container.ndarray([0.496714153011, -0.138264301171, 0.647688538101, 1.52302985641])))
        self.assertEqual(call_metadata.has_finished, True)
        self.assertEqual(call_metadata.iterations_done, None)

        self.assertIs(call_metadata.value.metadata.for_value, call_metadata.value)
        self.assertEqual(call_metadata.value.metadata.query((metadata.ALL_ELEMENTS,))['structural_type'], float)

    def test_metadata(self):
        expected_description = json.loads(EXPECTED_PRIMITIVE_DESCRIPTION_JSON)

        # We stringify to JSON and parse it to make sure the description can be stringified to JSON.
        description = json.loads(json.dumps(RandomPrimitive.metadata.to_json()))

        self.maxDiff = None
        self.assertEqual(expected_description, description)


if __name__ == '__main__':
    unittest.main()
