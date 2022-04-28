=======
History
=======

1.3.2
------------------
* force 'fork' for multiprocessing to ensure radiance state is copied to processes
* restructure radiancerrenderers - not singleton, just a stateful class, pickleable with get/set state
* dummy skydatamask class useful for intializing with lightresult axes to handle fill
* value_array method added to ResultAxis for easier syntax
* settable sigma_c method in hvsgsm
* make integrator.helpers public for overrides
* supress warnings from radiance during reflection search
* implement ZonalIntegratorDV

1.3.1 (2022-04-19)
------------------
* moved craytraverse to separate repository, now a requirement
* implemented glare sensation model, not yet available from CLI

1.3.0 (2022-04-01)
------------------
* first version compatible on linux systems
* changed skyres specification to int (defining side) for consistency with other resolution parameters

1.2.8 (2022-03-15)
------------------
* include radius around sun and reflections when resampling view. for 3comp, -ss should be 0 for skyengine
* handle stray hits when resampling radius around sun
* new simtype: 1compdv / integratordv

1.2.7 (2022-03-01)
------------------

* parametric search radius for specguide in sunsamplerpt
* integratorDS checks whether it is more memory efficient to apply skyvectors before adding points
* fixed double printing of 360 direct_views
* exposd lowlight and threshold parameter access to cli (both imgmetric and evaluate)
* changed to general precision formatting for lightresult printing
* fixed -skyfilter in pull, needs a skydata file to correctly index, otherwise based on array size
* new sampling metric normalizations, can now control logging and pbars with scene parameter

1.2.6 (2022-02-19)
------------------

* add hours when available to skydata
* proper masking of 360 images
* integratorDS handles stray roughness from direct patch
* planmapper, z set to median instead of max, added autorotation/alignment
* bugs/features/consistency in LightResult, need better usage documentation
* directviews from cli (only works with sky)

1.2.5 (2022-02-15)
------------------

* integrated zonal calcs in cli
* fall back to regular light result when possible (but keep area)
* fixed bugs in LightResult, ZonalLightResult
* added physically based point spread calculation that ~matches gregs gblur script, but using acutal lorentzian from reference
* added blur psf to sources in image evaluation


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
