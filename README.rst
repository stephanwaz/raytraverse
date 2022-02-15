======================
raytraverse (1.2.5)
======================

.. image:: https://img.shields.io/pypi/v/raytraverse?style=flat-square
    :target: https://pypi.org/project/raytraverse
    :alt: PyPI

.. image:: https://img.shields.io/pypi/l/raytraverse?style=flat-square
    :target: https://www.mozilla.org/en-US/MPL/2.0/
    :alt: PyPI - License

.. image:: https://img.shields.io/readthedocs/raytraverse/stable?style=flat-square
    :target: https://raytraverse.readthedocs.io/en/stable/
    :alt: Read the Docs (version)

.. image:: https://img.shields.io/travis/com/stephanwaz/raytraverse?style=flat-square
    :target: https://travis-ci.com/github/stephanwaz/raytraverse/builds
    :alt: Travis (.com)

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
* Documentation: https://raytraverse.readthedocs.io/en/latest/.


Installation
------------
The easiest way to install raytraverse is with pip::

    pip install --upgrade pip setuptools wheel
    pip install raytraverse

or if you have cloned this repository::

    cd path/to/this/file
    pip install .

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

Getting Started
---------------

the following example script shows the basic workflow for a complete simulation
it can be saved to a local file with::

    raytraverse examplescript > example.py

or the file is located at raytraverse/example.py

