=======
History
=======

1.2.4 (2021-12-03) (not posted until 2022-02-10)
------------------------------------------------

* organized command line code
* use process pool for sun sampler when raytracing is fast (such as -ab 0 runs with dcomp)
* propogate plotp to child sampler if sampling one level
* separated utility command line to own entry point. fixed ambiguity in
  coordinate handedness of some functions (changed kwarg defaults)

1.2.3 (2021-09-03)
------------------

 * fixed rcontrib to work with Radiance/HEAD, radiance version string includes commit
 * daylightplane - add indirect to -ab 0 sun run (daysim/5-phase style)
 * lightpointkd - handle adding points with same sample rays
 * sampler - add repeat function to follow an existing sampling scheme
 * lightresult - added print function
 * scene - remove logging from scene class
 * cli.py
    * new command imgmetric, extract rays from image and use same metricfuncs
    * mew command pull, filter and output 2d data frames from lightresult
    * add printdata option to suns, to see candidates or border
 * make TStqdm progress bar class public
 * include PositionIndex calculation in BaseMetricSet
     * new metrics: loggcr and position weighted luminance/gcr
 * skymapper: filter candidates by positive dirnorm when initialized with epw/wea
 * imagetools: parallel process image metrics, also normalize peak with some
    assumptions
 * lightresult: accept slices for findices argument
 * sunsamplerpt: at second and thrid sampling levels supplement sampling with
    spec_guide at 1/100 the threshold. helps with imterior spaces to find smaller
    patches of sun
 * positionindex: fix bug transcribed from evalglare with the positionindex below horizon


1.2.0/2 (2021-05-24)
--------------------
* command line interface development

1.1.2 (2021-02-19)
------------------
* improved documentation

1.1.0/1 (2021-02-10)
--------------------
* refactor code to operate on a single point at a time

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
