=======
History
=======

1.0.4 (2020-11-18)
------------------
* create and manage log file (attribute of Scene) for run directories
* possible fix for bug in interpolate_kd resulting in index range errors
* protect imports in cli.py so documentation can be built without installing

1.0.3 (2020-11-10)
------------------
* new module for calculating position based on retinal features
* view specifications for directview plotting
* options for samples/weight visibility on directview plotting

0.2.0 (2020-09-25)
------------------

* Build now includes all radiance dependencies to setup multi-platform testing
* In the absence of craytraverse, sampler falls back to SPRenderer
* install process streamlined for developer mode
* travis ci deploys linux and mac wheels directly to pypi
* release.sh should be run after updating this file, tests past locally and
    docs build.

0.1.0 (2020-05-19)
------------------

* First release on PyPI.
