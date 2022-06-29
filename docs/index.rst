.. include:: ../README.rst

Command Line Interface
----------------------

The raytraverse command provides command line access to executing common
tasks. The best way to manage all of the options is with a .cfg file. First,
generate a template::

    raytraverse --template > options.cfg

and then edit the options for each file. for example, here is a simplified
configuration for a low accuracy sample simulation, assuming a model scaled
in meters where plane.rad is betwee 4m and 10m on each side::

    [shared]
    weather_file = weather.epw

    [raytraverse_scene]
    out = outdir
    scene = room.rad

    [raytraverse_area]
    ptres = 2.0
    zone = plane.rad

    [raytraverse_suns]
    epwloc = True
    loc = ${shared:weather_file}

    [raytraverse_skydata]
    wea = ${shared:weather_file}
    skyres = 10

    [raytraverse_skyengine]
    accuracy = 2.0
    skyres = 10

    [raytraverse_sunengine]
    accuracy = 2.0
    rayargs = -ab 0
    nlev = 5

    [raytraverse_skyrun]
    accuracy = 2.0
    edgemode = reflect
    nlev = 2

    [raytraverse_sunrun]
    accuracy = 3.0
    edgemode = reflect
    nlev = 2
    srcaccuracy = 2.0
    srcnlev = 2

    [raytraverse_images]
    basename = results
    blursun = True
    interpolate = highc
    res = 800
    resampleview = True
    sdirs = None
    sensors = None
    skymask = 0:24

    [raytraverse_evaluate]
    basename = results
    sdirs = None
    sensors = None
    skymask = None

    [raytraverse_pull]
    col = metric point
    gridhdr = True
    ofiles = results
    skyfill = ${shared:weather_file}
    viewfilter = 0

and then from the command line run::

    raytraverse -c options.cfg skyrun directskyrun sunrun evaluate pull


.. toctree::
   :maxdepth: 2
   :caption: Command Line Interface

   cli


.. toctree::
   :caption: API
   :maxdepth: 2
   :hidden:

   scene
   mapper
   formatter
   renderer
   sky

   sampler
   lightpoint
   lightfield
   integrator
   evaluate
   craytraverse

   io
   translate
   utility
   api

Tutorials
---------

.. toctree::
    :caption: Toturials
    :maxdepth: 1

    simaud2021

.. toctree::
   :maxdepth: 3
   :hidden:
   :caption: Navigation

   history
   genindex
   search

.. include:: ../ACKNOWLEDGEMENTS.rst
