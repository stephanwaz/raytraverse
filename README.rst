===========
raytraverse
===========


.. image:: https://img.shields.io/pypi/v/raytraverse.svg
        :target: https://pypi.python.org/pypi/raytraverse
		:alt: Release Status

.. image:: https://readthedocs.org/projects/raytraverse/badge/?version=latest
        :target: https://raytraverse.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status
		
.. image:: https://travis-ci.org/stephanwaz/raytraverse.svg
	:target: https://travis-ci.org/stephanwaz/raytraverse
	:alt: Build Status

raytraverse is a complete workflow for climate based daylight modelling,
simulation, and evaluation of architectural spaces. Built around a variance
based adaptive sampling strategy, raytraverse can fully explore the daylight
conditions throughout a space with efficient use of processing power and
storage space.

* Free software: Mozilla Public License 2.0 (MPL 2.0)
* Documentation: https://raytraverse.readthedocs.io.


Installation
------------
The easiest way to install raytraverse is with pip::

    pip install raytraverse

or if you have cloned this repository::

    cd path/to/this/file
    pip install .

note that on first run one of the required modules will download some auxilary
data which could take a minute, after that first run start-up is much faster.

Usage
-----
raytraverse includes a complete command line interface with all commands
nested under the `raytraverse` parent command enter::

    raytraverse --help

raytraverse also exposes an object oriented API written primarily in python.
calls to Radiance are made through Renderer objects that wrap the radiance
c source code in c++ classes, which are made available in python with pybind11.
see the src/ directory for more.

For complete documentation of the API and the command line interface either
use the Documentation link included above or::

    pip install -r requirements_dev.txt
    make docs

to generate local documentation.

Git Stuff
---------
this project is hosted in too places, a private repo (master branch) at:

	https://gitlab.enterpriselab.ch/lightfields/raytraverse

and a public repo (release branch) at:

	https://github.com/stephanwaz/raytraverse

the repo also depends on two submodules, to initialize run the following::

	git clone https://github.com/stephanwaz/raytraverse
	cd raytraverse
	git submodule init
	git submodule update --remote
	git -C src/Radiance config core.sparseCheckout true 
	cp src/sparse-checkout .git/modules/src/Radiance/info/
	git submodule update --remote --force src/Radiance
	
after a "git pull" make sure you also run::

	git submodule update
	
to track with the latest commit used by raytraverse.

Licence
-------

| Copyright (c) 2020 Stephen Wasilewski
| This Source Code Form is subject to the terms of the Mozilla Public
| License, v. 2.0. If a copy of the MPL was not distributed with this
| file, You can obtain one at http://mozilla.org/MPL/2.0/.

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

