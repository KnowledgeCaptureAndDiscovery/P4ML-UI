import os
import sys
from setuptools import setup, find_packages

PACKAGE_NAME = 'primitive_interfaces'
MINIMUM_PYTHON_VERSION = 3, 6


def check_python_version():
    """Exit when the Python version is too low."""
    if sys.version_info < MINIMUM_PYTHON_VERSION:
        sys.exit("Python {}.{}+ is required.".format(*MINIMUM_PYTHON_VERSION))


def read_package_variable(key):
    """Read the value of a variable from the package without importing."""
    module_path = os.path.join(PACKAGE_NAME, '__init__.py')
    with open(module_path) as module:
        for line in module:
            parts = line.strip().split(' ')
            if parts and parts[0] == key:
                return parts[-1].strip("'")
    assert False, "'{0}' not found in '{1}'".format(key, module_path)


check_python_version()
version = read_package_variable('__version__')

setup(
    name=PACKAGE_NAME,
    version=version,
    description='Python interfaces for TA1 primitives',
    author='DARPA D3M Program',
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    install_requires=[
        'd3m_metadata=={version}'.format(version=version),
    ],
    tests_require=[
        'docker',
    ],
    dependency_links=[
        'git+https://gitlab.com/datadrivendiscovery/metadata.git@devel#egg=d3m_metadata-{version}'.format(version=version),
    ],
    url='https://gitlab.com/datadrivendiscovery/primitive-interfaces',
)
