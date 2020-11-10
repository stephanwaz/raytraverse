===================
raytraverse (1.0.3)
===================

.. image:: https://img.shields.io/pypi/v/raytraverse?style=flat-square
    :target: https://pypi.org/project/raytraverse
    :alt: PyPI

.. image:: https://img.shields.io/pypi/l/raytraverse?style=flat-square
    :target: https://www.mozilla.org/en-US/MPL/2.0/
    :alt: PyPI - License

.. image:: https://img.shields.io/readthedocs/raytraverse/stable?style=flat-square
    :target: https://raytraverse.readthedocs.io/en/stable/
    :alt: Read the Docs (version)

.. image:: https://img.shields.io/travis/stephanwaz/raytraverse?style=flat-square
    :target: https://travis-ci.org/github/stephanwaz/raytraverse/builds
    :alt: Travis (.org)

.. image:: https://img.shields.io/coveralls/github/stephanwaz/raytraverse?style=flat-square
    :target: https://coveralls.io/github/stephanwaz/raytraverse
    :alt: Coveralls github

.. image:: https://zenodo.org/badge/doi/10.5281/zenodo.4091318.svg
   :target: https://zenodo.org/badge/latestdoi/296295567

raytraverse is a complete workflow for climate based daylight modelling,
simulation, and evaluation of architectural spaces. Built around a wavelet
guided adaptive sampling strategy, raytraverse can fully explore the daylight
conditions throughout a space with efficient use of processing power and
storage space.

* Free software: Mozilla Public License 2.0 (MPL 2.0)
* Documentation: https://raytraverse.readthedocs.io/en/stable/.


Installation
------------
The easiest way to install raytraverse is with pip::

    pip install --upgrade pip setuptools wheel
    pip install raytraverse

or if you have cloned this repository::

    cd path/to/this/file
    pip install .


note that on first run the skycalc module may download some auxilary
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
this project is hosted in two places, a private repo (master branch) at:

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

Acknowledgements
----------------

Thanks to additional project collaborators and advisors Marilyne Andersen, Lars
Grobe, Roland Schregle, Jan Wienold, and Stephen Wittkopf

This software development was financially supported by the Swiss National
Science Foundation as part of the ongoing research project “Light fields in
climate-based daylight modeling for spatio-temporal glare assessment”
(SNSF_ #179067).

Software Credits
----------------

    - Raytraverse uses Radiance_
    - As well as all packages listed in the requirements.txt file,
      raytraverse relies heavily on the Python packages numpy_, scipy_, and
      pywavelets_ for key parts of the implementation.
    - C++ bindings, including exposing core radiance functions as methods to
      the renderer classes are made with pybind11_
    - Installation and building from source uses cmake_ and scikit-build_
    - This package was created with Cookiecutter_ and the
      `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
.. _Radiance: https://www.radiance-online.org
.. _numpy: https://numpy.org/doc/stable/reference/
.. _scipy: https://docs.scipy.org/doc/scipy/reference/
.. _pywavelets: https://pywavelets.readthedocs.io/en/latest/
.. _pybind11: https://pybind11.readthedocs.io/en/stable/index.html
.. _scikit-build: https://scikit-build.readthedocs.io/en/latest/
.. _SNSF: http://www.snf.ch/en/Pages/default.aspx
.. _cmake: https://cmake.org/cmake/help/latest/
