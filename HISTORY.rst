=======
History
=======

1.3.9 (2022-12-08)
------------------
* improvements to more easily load lightpoints directly from files without context
* api.load_config for better compatibility between scripting/CLI
* change eccentricity model in hvsgsm
* fix bug in autorotate in planmapper when result should be 0 or 90

1.3.8 (2022-11-04)
------------------
* 0 variance bug in image interpolation
* added boundaries to LightResults
* accept dew_point in ttsv format to skydata
* outtput skydata with dewpoint using skydata_dew
* pull gridhdr no longer needs zone if it is embedded in LightResult

1.3.7 (2022-10-26)
------------------
* updated resonse fit in hsvgm
* resolution option for pull 2 hdr cli
* modules directly available from import raytraverse
* ensure parameters set correctly so sun is always resampled in 1compdv
* bug in integrator log showing too many chunks
* rebase method added to basic LightResult
* rewrote lightpoint image interpolation
* SrcSamplerPt no longer uses accuracy, instead, set t0, at t1 with physical units
* t0 and t1 now instance properties (settable from init)
* added direct view options to raytu lp2img (warning, non-fisheye color throws error)
* preserve -ss in direct view sampling
* clean up srcview sampling (always distant) and fix double counting image when rough specular gets re-samples
* new factoring with craytraverse / renderer inheritance fixes rcontrib reset

1.3.6 (2022-07-28)
------------------
* add scale to sensor integrator forr proper unit conversion (lux by default)
* parallel processing in zonallightresult.pull2hdr
* add lightresult.pull2planhdr to match signature of zonallightresult
* add zonallightresult.rebase to make standard lightresult from zonal
* fixed bug in sunsplanekd.query_by_sun that returned all points, not just best matches
* added index() function to resultaxis
* bug fixes in sensorintegrator, needed additional function overrides and index broadcasting
* avoid IndexError at the end of skydata.maskindices
* add lightresult.merge (and cli interface with raytu merge) for combining LightResults
* change chunking of large calls to evaluate for better performance and the save intermediate results
* pass jitterrate to MaskedPlanMapper constructor
* rewrote RadianceFormatter.get_scene() parser, not based on file extensions
* bug in SamplerArea when operating with MaskedPlanMapper, possible to have
  no samples, leading to IndexError, fixed at self._mask initialization so
  atleast one cell is True.
* added gss to raytu imgmetric (no options yet, uses standard observer)


1.3.5 (2022-07-05)
------------------
* better memory management in zonal sensorintegrator
* plot each weight in srcsamplerpt when using detail/color
* slight reorganization in Integrator to accommodate sensorintegrator changes
* fixed bug in pull with -skyfilter but no -skyfill
* allow skydata write without scene
* change default sunrun parameter to -ab 0
* updated installation instructions and Dockerfiles to include radiance installation
* added adpatch for better control over default args in Rcontrib
* 2x speedup in translate.calc_omega by checking for containment before intersection
  left commented code for pygeos method, but it is slower without better way to
  read in voronoi (creation with pygeos only uses small fraction of points).
* formatting change in CLI docstrings to avoid error with latest docutils

1.3.4 (2022-06-21)
------------------
* do not use srcview for local light sources, include atleast 1 level of clean-up
* make sure kd tree is rebuilt when lucky squirrel
* ambient file handling in rtrace
* better memory management in reflection_search (still a problem?)
* new example config with proper settings
* with minsamp > 0 make sure from_pdf returns something so sampling can complete

1.3.3 (2022-06-07)
------------------
* static light source sampler, directly samples electric lights at appropriate level,
  will use lots of extra samples with very long thin fixtures
* color support in lightPointKD and samplers, but for now only works with imagesampler and
  sourcesampler because need to update skydata to work with color (and handle mixed data)
* use scene detail in sampler (in this case image reconstruction works better WITHOUT
  scene detail, new interpolation keywords fastc and highc for context interpolation)
* consolidated integrator/zonalintegrator and special methods dv/ds into one class
* changed zonal sunplane query algorithm: filter suns, penalize, query instead of filter suns, sort, filter points
* removed ptfilter keyword for zonal evaluation (new process does not use)
* sunplane normalization based on level 0 distance of sampled suns and level 0 distance of areas
  for level 0 sampled suns
* SensorIntegrator to process sensorplane results
* manage stranded open OS file descriptors
* wait to calculate omega on demand in lightplaneKD
* removed img2lf in imagetools, creates circular reference, need to add to different module
* allow None vector argument for lightplane initialization (cconstructs filename)
* zero pad hour labels in lightresult for proper file name sorting
* calc_omega method now passes "QJ" to qhull which seems to reliable return regions for all points
  in case of failure, distributed area among points sharing region (moved from integrator.helpers to translate)
  so LightPointKD can share
* fixed mistakes in GSS implementation and recalibrated

1.3.2 (2022-04-28)
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
