===================
raytraverse (1.0.2)
===================

.. image:: https://img.shields.io/pypi/v/raytraverse?style=flat-square
    :target: https://pypi.org/project/raytraverse/1.0.2
    :alt: PyPI

.. image:: https://img.shields.io/pypi/l/raytraverse?style=flat-square
    :target: https://www.mozilla.org/en-US/MPL/2.0/
    :alt: PyPI - License

.. image:: https://img.shields.io/readthedocs/raytraverse/v1.0.2?style=flat-square
    :target: https://raytraverse.readthedocs.io/en/v1.0.2/
    :alt: Read the Docs (version)

.. image:: https://img.shields.io/travis/stephanwaz/raytraverse/v1.0.2?style=flat-square
    :target: https://travis-ci.org/github/stephanwaz/raytraverse/builds
    :alt: Travis (.org)

.. image:: https://img.shields.io/coveralls/github/stephanwaz/raytraverse/v1.0.2?style=flat-square
    :target: https://coveralls.io/github/stephanwaz/raytraverse
    :alt: Coveralls github

.. image:: https://zenodo.org/badge/296295567.svg
   :target: https://zenodo.org/badge/latestdoi/296295567

raytraverse is a complete workflow for climate based daylight modelling,
simulation, and evaluation of architectural spaces. Built around a variance
based adaptive sampling strategy, raytraverse can fully explore the daylight
conditions throughout a space with efficient use of processing power and
storage space.

* Free software: Mozilla Public License 2.0 (MPL 2.0)
* Documentation: https://raytraverse.readthedocs.io/en/v1.0.2/.


Installation
------------
The easiest way to install raytraverse is with pip::

    pip install --upgrade pip setuptools wheel
    pip install raytraverse

or if you have cloned this repository::

    cd path/to/this/file
    pip install .


note that on first run one of the required modules may download some auxilary
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

    pip install -r docs/requirements.txt
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

| Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
| This Source Code Form is subject to the terms of the Mozilla Public
| License, v. 2.0. If a copy of the MPL was not distributed with this
| file, You can obtain one at http://mozilla.org/MPL/2.0/.

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

