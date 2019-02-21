import setuptools.command.build_py
import distutils.cmd
import distutils.log
import setuptools
import subprocess
import unittest
import os

class pipInstallCommand(distutils.cmd.Command):
    """A custom command to install pip dependencies in proper order."""
    
    description = 'install pip dependencies'
    user_options = [
        # The format is (long option, short option, description).
        ('develop', 'e', 'install in develop mode'),
    ]
    boolean_options = ['develop']
    
    def initialize_options(self):
        """Set default values for options."""
        # Each user option must be listed here with their default value.
        self.develop = False
    
    
    def finalize_options(self):
        """Post-process options."""
    
    def run(self):
        """Run command."""
        thisdir = os.path.dirname(os.path.abspath(__file__))
        if self.develop:
            c = 'pip install --upgrade -e '
        else:
            c = 'pip install --upgrade '
        c1 = c + thisdir
        c2 = 'pip install librosa==0.5.1'
        c3 = 'pip install cvxpy==0.4.9'
        
        self.announce(
              'Running command: %s' % str(c1),
              level=distutils.log.INFO)
        subprocess.check_call(c1, shell=True)
        
        self.announce(
              'Running command: %s' % str(c2),
              level=distutils.log.INFO)
        subprocess.check_call(c2, shell=True)

        self.announce(
              'Running command: %s' % str(c3),
              level=distutils.log.INFO)
        subprocess.check_call(c3, shell=True)
    
# class buildPyCommand(setuptools.command.build_py.build_py):
#     """Custom build command."""
#     
#     def run(self):
#         self.run_command('pip')
#         setuptools.command.build_py.build_py.run(self)

setuptools.setup(
    name="spider",
    version="0.0.3",
    author="Jason Corso, Laura Balzano and The University of Michigan DARPA D3M Spider Team",
    author_email="jjcorso@umich.edu,girasole@umich.edu,davjoh@umich.edu",
    url="https://gitlab.datadrivendiscovery.org/michigan/spider",
    license="MIT",
    description="DARPA D3M Spider Project Code",
    install_requires=[
        "numpy (>=1.12.1)",
        "scipy (>=0.19.0)",
        "scikit-learn (>=0.18.1)",
        "matplotlib (>=1.5.1)",
        "Pillow (>=4.1.1)",
        "h5py (>=2.7.0)",
        "opencv-python (>=3.0.0)",
        "keras (>=2.0.4)",
        "tensorflow (>=1.1.0)",
        "pandas (>=0.19.2)",
        "typing (>=3.6.2)",
        "stopit (>=1.1.1)"
    ],
    packages=["spider",
                "spider.featurization",
                "spider.featurization.vgg16", 
                "spider.featurization.audio_featurization",
                "spider.featurization.logmelspectrogram",
                "spider.featurization.audio_slicer",
                "spider.distance",
                "spider.distance.rfd",
                "spider.cluster",
                "spider.cluster.ssc_cvx",
                "spider.cluster.ssc_admm",
                "spider.cluster.kss",
                "spider.cluster.ekss",
                "spider.cluster.ssc_omp",
                "spider.dimensionality_reduction",
                "spider.dimensionality_reduction.pcp_ialm",
                "spider.dimensionality_reduction.go_dec",
                "spider.dimensionality_reduction.rpca_lbd"],
    #scripts    =[],
    #ext_modules=[],
    cmdclass={
              'pip': pipInstallCommand,
#               'build_py': buildPyCommand,
              },
    #entry_points = {
    #    'console_scripts': [
    #        'spidermaketestdata=spider.tests.maketestdata:main'
    #    ],
    #    'gui_scripts': [
    #        'spiderloss=spider.monitors.loss:main'
    #    ],
    #},
    test_suite='spider.tests.suite'
)
