import typing
import unittest

from d3m_metadata import container, hyperparams, metadata as metadata_module, params, utils
from primitive_interfaces import base, transformer


class TestMetadata(unittest.TestCase):
    # Test more complicated hyper parameters.
    def test_hyperparms(self):
        Inputs = container.List[float]
        Outputs = container.List[float]

        class Hyperparams(hyperparams.Hyperparams):
            n_components = hyperparams.Hyperparameter[typing.Union[int, None]](
                default=None,
                description='Number of components (< n_classes - 1) for dimensionality reduction.',
            )
            learning_rate = hyperparams.Uniform(
                lower=0.01,
                upper=2,
                default=0.1,
                description='Learning rate shrinks the contribution of each classifier by ``learning_rate``. There is a trade-off between ``learning_rate`` and ``n_estimators``.',
            )

        class TestPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
            metadata = metadata_module.PrimitiveMetadata({
                'id': '67568a80-dec2-4597-a10f-39afb13d3b9c',
                'version': '0.1.0',
                'name': "Test Primitive",
                'source': {
                    'name': 'Test',
                },
                'python_path': 'd3m.primitives.test.TestPrimitive',
                'algorithm_types': [
                    metadata_module.PrimitiveAlgorithmType.NUMERICAL_METHOD,
                ],
                'primitive_family': metadata_module.PrimitiveFamily.OPERATOR,
            })

            def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
                pass

        self.assertEqual(TestPrimitive.metadata.query()['primitive_code']['hyperparams'], {
            'n_components': {
                'type': hyperparams.Hyperparameter,
                'default': 'None',
                'structural_type': typing.Union[int, type(None)],
                'semantic_types': (),
                'description': 'Number of components (< n_classes - 1) for dimensionality reduction.',
            },
            'learning_rate': {
                'type': hyperparams.Uniform,
                'default': 0.1,
                'structural_type': float,
                'semantic_types': (),
                'description': 'Learning rate shrinks the contribution of each classifier by ``learning_rate``. There is a trade-off between ``learning_rate`` and ``n_estimators``.',
                'lower': 0.01,
                'upper': 2,
                'upper_inclusive': False,
            },
        })

    def test_package_validation(self):
        Inputs = container.List[float]
        Outputs = container.List[float]

        class Hyperparams(hyperparams.Hyperparams):
            pass

        with self.assertRaisesRegex(ValueError, 'Invalid package name'):
            class TestPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
                metadata = metadata_module.PrimitiveMetadata({
                    'id': '67568a80-dec2-4597-a10f-39afb13d3b9c',
                    'version': '0.1.0',
                    'name': "Test Primitive",
                    'source': {
                        'name': 'Test',
                    },
                    'installation': [{
                        'type': metadata_module.PrimitiveInstallationType.PIP,
                        'package': 'git+https://gitlab.com/datadrivendiscovery/tests-data.git',
                        'version': '0.1.0',
                    }],
                    'python_path': 'd3m.primitives.test.TestPrimitive',
                    'algorithm_types': [
                        metadata_module.PrimitiveAlgorithmType.NUMERICAL_METHOD,
                    ],
                    'primitive_family': metadata_module.PrimitiveFamily.OPERATOR,
                })

                def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
                    pass

    def test_package_uri_validation(self):
        Inputs = container.List[float]
        Outputs = container.List[float]

        class Hyperparams(hyperparams.Hyperparams):
            pass

        with self.assertRaisesRegex(ValueError, 'Package URI does not include a commit hash'):
            class TestPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
                metadata = metadata_module.PrimitiveMetadata({
                    'id': '67568a80-dec2-4597-a10f-39afb13d3b9c',
                    'version': '0.1.0',
                    'name': "Test Primitive",
                    'source': {
                        'name': 'Test',
                    },
                    'installation': [{
                        'type': metadata_module.PrimitiveInstallationType.PIP,
                        'package_uri': 'git+https://gitlab.com/datadrivendiscovery/tests-data.git',
                    }],
                    'python_path': 'd3m.primitives.test.TestPrimitive',
                    'algorithm_types': [
                        metadata_module.PrimitiveAlgorithmType.NUMERICAL_METHOD,
                    ],
                    'primitive_family': metadata_module.PrimitiveFamily.OPERATOR,
                })

                def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
                    pass

        with self.assertRaisesRegex(ValueError, 'Package URI does not include a commit hash'):
            class TestPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
                metadata = metadata_module.PrimitiveMetadata({
                    'id': '67568a80-dec2-4597-a10f-39afb13d3b9c',
                    'version': '0.1.0',
                    'name': "Test Primitive",
                    'source': {
                        'name': 'Test',
                    },
                    'installation': [{
                        # Once with string.
                        'type': 'PIP',
                        'package_uri': 'git+https://gitlab.com/datadrivendiscovery/tests-data.git@v0.1.0',
                    }],
                    'python_path': 'd3m.primitives.test.TestPrimitive',
                    'algorithm_types': [
                        metadata_module.PrimitiveAlgorithmType.NUMERICAL_METHOD,
                    ],
                    'primitive_family': metadata_module.PrimitiveFamily.OPERATOR,
                })

                def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
                    pass

        with self.assertRaisesRegex(ValueError, 'Package URI does not include a commit hash'):
            class TestPrimitive(transformer.TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
                metadata = metadata_module.PrimitiveMetadata({
                    'id': '67568a80-dec2-4597-a10f-39afb13d3b9c',
                    'version': '0.1.0',
                    'name': "Test Primitive",
                    'source': {
                        'name': 'Test',
                    },
                    'installation': [{
                        # Once with enum value.
                        'type': metadata_module.PrimitiveInstallationType.PIP,
                        'package_uri': 'git+https://gitlab.com/datadrivendiscovery/tests-data.git@v0.1.0',
                    }],
                    'python_path': 'd3m.primitives.test.TestPrimitive',
                    'algorithm_types': [
                        metadata_module.PrimitiveAlgorithmType.NUMERICAL_METHOD,
                    ],
                    'primitive_family': metadata_module.PrimitiveFamily.OPERATOR,
                })

                def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> base.CallResult[Outputs]:
                    pass


if __name__ == '__main__':
    unittest.main()
