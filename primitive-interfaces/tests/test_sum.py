import json
import unittest
import os
import os.path
import sys
import time

import docker

import primitive_interfaces
from d3m_metadata import container, metadata, utils

TEST_PRIMITIVES_DIR = os.path.join(os.path.dirname(__file__), 'data', 'primitives')

sys.path.insert(0, TEST_PRIMITIVES_DIR)

from test_primitives.sum import SumPrimitive


EXPECTED_PRIMITIVE_DESCRIPTION_JSON = r"""
{
    "id": "9c00d42d-382d-4177-a0e7-082da88a29c8",
    "version": "0.1.0",
    "name": "Sum Values",
    "keywords": [
        "test primitive"
    ],
    "source": {
        "name": "Test team",
        "uris": [
            "https://gitlab.com/datadrivendiscovery/tests-data/blob/master/primitives/test_primitives/sum.py",
            "https://gitlab.com/datadrivendiscovery/tests-data.git"
        ]
    },
    "installation": [
        {
            "type": "PIP",
            "package_uri": "git+https://gitlab.com/datadrivendiscovery/tests-data.git@__GIT_COMMIT__#egg=test_primitives&subdirectory=primitives"
        },
        {
            "type": "DOCKER",
            "key": "summing",
            "image_name": "registry.gitlab.com/datadrivendiscovery/tests-data/summing",
            "image_digest": "sha256:07db5fef262c1172de5c1db5334944b2f58a679e4bb9ea6232234d71239deb64"
        }
    ],
    "location_uris": [
        "https://gitlab.com/datadrivendiscovery/tests-data/raw/__GIT_COMMIT__/primitives/test_primitives/sum.py"
    ],
    "python_path": "d3m.primitives.test.SumPrimitive",
    "algorithm_types": [
        "COMPUTER_ALGEBRA"
    ],
    "primitive_family": "OPERATOR",
    "preconditions": [
        "NO_MISSING_VALUES",
        "NO_CATEGORICAL_VALUES"
    ],
    "schema": "https://metadata.datadrivendiscovery.org/schemas/v0/primitive.json",
    "original_python_path": "test_primitives.sum.SumPrimitive",
    "primitive_code": {
        "class_type_arguments": {
            "Inputs": "typing.Union[d3m_metadata.container.numpy.ndarray, d3m_metadata.container.list.List[float], d3m_metadata.container.list.List[d3m_metadata.container.list.List[float]]]",
            "Outputs": "d3m_metadata.container.list.List[float]",
            "Hyperparams": "test_primitives.sum.Hyperparams",
            "Params": "NoneType"
        },
        "interfaces_version": "__INTERFACES_VERSION__",
        "interfaces": [
            "base.SingletonOutputMixin",
            "transformer.TransformerPrimitiveBase",
            "base.PrimitiveBase"
        ],
        "hyperparams": {},
        "arguments": {
            "hyperparams": {
                "type": "test_primitives.sum.Hyperparams",
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
                "type": "typing.Union[d3m_metadata.container.numpy.ndarray, d3m_metadata.container.list.List[float], d3m_metadata.container.list.List[d3m_metadata.container.list.List[float]]]",
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
                "returns": "primitive_interfaces.base.CallResult[d3m_metadata.container.list.List[float]]"
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
    "structural_type": "test_primitives.sum.SumPrimitive",
    "description": "A primitive which sums all the values on input into one number."
}
""".replace('__INTERFACES_VERSION__', primitive_interfaces.__version__).replace('__GIT_COMMIT__', utils.current_git_commit(TEST_PRIMITIVES_DIR))


class TestSumPrimitive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docker_client = docker.from_env()

        cls.docker_containers = {}

        # Start all containers (this pulls images if they do not yet exist).
        installation = SumPrimitive.metadata.query().get('installation', [])
        for entry in installation:
            if entry['type'] != metadata.PrimitiveInstallationType.DOCKER:
                continue

            image = cls.docker_client.images.pull('{image_name}@{image_digest}'.format(image_name=entry['image_name'], image_digest=entry['image_digest']))
            exposed_ports = image.attrs['Config']['ExposedPorts']

            cls.docker_containers[entry['key']] = cls.docker_client.containers.run(
                '{image_name}@{image_digest}'.format(image_name=entry['image_name'], image_digest=entry['image_digest']),
                # Ports have to be mapped to the host so that they works in GitLab CI and Docker-in-Docker
                # environment (ports are mapped to the Docker-in-Docker container itself, not the real host).
                detach=True, auto_remove=True, ports={port: port for port in exposed_ports.keys()},
            )

        # Wait a bit for things to run. Even if status is "running" it does
        # not really mean all services inside are really already running.
        time.sleep(1)  # 1 s

        # Wait for containers to be running.
        for container in cls.docker_containers.values():
            for _ in range(100):  # 100 * 100 ms = 10 s
                container.reload()
                if container.status == 'running':
                    assert container.attrs.get('NetworkSettings', {}).get('IPAddress', None)
                    break
                elif container.status in ('removing', 'paused', 'exited', 'dead'):
                    raise ValueError("Container '{container}' is not running.".format(container=container))

                time.sleep(0.1)  # 100 ms
            else:
                raise ValueError("Container '{container}' is not running.".format(container=container))

    @classmethod
    def tearDownClass(cls):
        for key, container in cls.docker_containers.items():
            container.stop()

        cls.docker_containers = {}

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

    def get_docker_containers(self):
        if os.environ.get('GITLAB_CI', None):
            # In GitLab CI we use Docker-in-Docker to run containers, so container's port is available on Docker-in-Docker
            # container itself (with hostname "docker") and we cannot directly connect to the primitive's container.
            return {key: 'docker' for key, container in self.docker_containers.items()}
        else:
            return {key: container.attrs['NetworkSettings']['IPAddress'] for key, container in self.docker_containers.items()}

    def test_ndarray(self):
        hyperparams_class = SumPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']

        primitive = SumPrimitive(hyperparams=hyperparams_class.defaults(), docker_containers=self.get_docker_containers())

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

        # Because it is a singleton primitive we can know that there is exactly one value in outputs.
        result = call_metadata.value[0]

        self.assertEqual(result, 36)
        self.assertEqual(call_metadata.has_finished, True)
        self.assertEqual(call_metadata.iterations_done, None)

        self.assertIs(call_metadata.value.metadata.for_value, call_metadata.value)
        self.assertEqual(call_metadata.value.metadata.query((metadata.ALL_ELEMENTS,))['structural_type'], float)

    def test_lists(self):
        hyperparams_class = SumPrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']

        primitive = SumPrimitive(hyperparams=hyperparams_class.defaults(), docker_containers=self.get_docker_containers())

        inputs = container.List[container.List[float]]([container.List[float]([1, 2, 3, 4]), container.List[float]([5, 6, 7, 8])], {
            'schema': metadata.CONTAINER_SCHEMA_VERSION,
            'structural_type': container.List[container.List[float]],
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
            'structural_type': float,
        })

        call_metadata = self.call_primitive(primitive, 'produce', inputs=inputs)

        # Because it is a singleton primitive we can know that there is exactly one value in outputs.
        result = call_metadata.value[0]

        self.assertEqual(result, 36)
        self.assertEqual(call_metadata.has_finished, True)
        self.assertEqual(call_metadata.iterations_done, None)

        self.assertIs(call_metadata.value.metadata.for_value, call_metadata.value)
        self.assertEqual(call_metadata.value.metadata.query((metadata.ALL_ELEMENTS,))['structural_type'], float)

    def test_can_accept(self):
        inputs_metadata = metadata.DataMetadata({
            'schema': metadata.CONTAINER_SCHEMA_VERSION,
            'structural_type': container.ndarray,
        })

        self.assertFalse(SumPrimitive.can_accept(method_name='produce', arguments={
            'inputs': inputs_metadata,
        }))

        inputs_metadata.update((), {
            'dimension': {
                'length': 2,
            },
        })

        self.assertFalse(SumPrimitive.can_accept(method_name='produce', arguments={
            'inputs': inputs_metadata,
        }))

        inputs_metadata.update((metadata.ALL_ELEMENTS,), {
            'dimension': {
                'length': 2,
            },
        })

        self.assertFalse(SumPrimitive.can_accept(method_name='produce', arguments={
            'inputs': inputs_metadata,
        }))

        inputs_metadata.update((metadata.ALL_ELEMENTS, metadata.ALL_ELEMENTS), {
            'structural_type': str,
        })

        self.assertFalse(SumPrimitive.can_accept(method_name='produce', arguments={
            'inputs': inputs_metadata,
        }))

        inputs_metadata.update((metadata.ALL_ELEMENTS, metadata.ALL_ELEMENTS), {
            'structural_type': float,
        })

        self.assertFalse(SumPrimitive.can_accept(method_name='produce', arguments={
            'inputs': inputs_metadata,
        }))

    def test_metadata(self):
        expected_description = json.loads(EXPECTED_PRIMITIVE_DESCRIPTION_JSON)

        # We stringify to JSON and parse it to make sure the description can be stringified to JSON.
        description = json.loads(json.dumps(SumPrimitive.metadata.to_json()))

        self.maxDiff = None
        self.assertEqual(expected_description, description)


if __name__ == '__main__':
    unittest.main()
