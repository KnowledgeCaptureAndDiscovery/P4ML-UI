spider -- The University of Michigan DARPA D3M SPIDER Project CodeBase
=================================

This codebase includes all methods and primitives that comprise the SPIDER 
project.  These do include the featurization primitives that we contributed as 
well as the featurization baseclass that provides a standard protocol for 
accessing and implementing the featurization across the D3M program.


Maintained by Jason Corso, Laura Balzano and DARPA D3M SPIDER project team members
at the University of Michigan.

Brief Description and File System Layout

    spider/                     --->  main package
      featurization/            --->  sub-package for featurization primitives and examples
      distance/                 --->  sub-package for distance measurement primitives and examples
      cluster/                  --->  sub-package for clustering primitives
      dimensionality_reduction/ --->  sub-package for dimensionality reduction primitives
      tests/                    --->  sub-package with unit tests 
$     primitives/               --->  sub-package for spider primitives
$     deeplayers/               --->  sub-package for new deep learning layers (that can 
                                        be integrated into deep-net primitives)
                                 
$ means currently empty and planned

Primitives Included

    # Featurization Primitives
    spider.featurization.vgg16
    spider.featurization.audio
    spider.featurization.audo_slicer
    spider.featurization.logmelspectrogram

    # Distance Primitives
    spider.distance.rfd

    # Cluster Primitives
    spider.cluster.kss
    spider.cluster.ekss
    spider.cluster.ssc_cvx
    spider.cluster.ssc_admm
    spider.cluster.ssc_omp

    # Dimensionality Reduction Primitives
    spider.dimensionality_reduction.pcp_ialm
    spider.dimensionality_reduction.go_dec
    spider.dimensionality_reduction.rpca_lbd

Executables Created

    spider/distance/examples/rfd.py
    spider/featurization/examples/vgg16.py
    spider/featurization/examples/audio.py
    spider/featurization/examples/audio_slicer.py
    spider/featurization/examples/logmelspectrogram.py
    spider/featurization/examples/train_audio.py

License
-------

MIT license.

Setup
-----

To begin working with this project, clone the repository to your machine.  

    git clone git@gitlab.datadrivendiscovery.org:michigan/spider.git


To let pip build and install this and any dependencies:

First run:

    pip --upgrade -r requirements.txt

Then:
Normal Install:
    
    python setup.py pip

Or Developer Mode:

    python setup.py pip -e

Then, to run unit tests:

    python setup.py test


Uninstall
---------

If you have pip installed, it is easy to uninstall spider

    pip uninstall spider

