.. include:: ../README.rst

Command Line Interface
----------------------

The raytraverse command provides command line access to executing common
tasks. The best way to manage all of the options is with a .cfg file. First,
generate a template::

    raytraverse --template > options.cfg

and then edit the options for each file. for example::

    [raytraverse_scene]
    out = outdir
    scene = room.rad

    [raytraverse_area]
    ptres = 2.0
    zone = plane.rad

    [raytraverse_suns]
    loc = weather.epw
    epwloc = True

    [raytraverse_skydata]
    wea = weather.epw

    [raytraverse_skyengine]
    accuracy = 2.0
    rayargs = -ab 2 -ad 4 -c 1000

    [raytraverse_sunengine]
    accuracy = 2.0
    rayargs = -ab 2 -c 1

    [raytraverse_skyrun]
    accuracy = 2.0
    jitter = True
    nlev = 2
    overwrite = False

    [raytraverse_sunrun]
    accuracy = 2.0
    nlev = 2
    srcaccuracy = 2.0

and then from the command line run::

    raytraverse -c options.cfg skyrun sunrun


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
