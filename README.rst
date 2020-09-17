===========
raytraverse
===========


.. image:: https://img.shields.io/pypi/v/raytraverse.svg
        :target: https://pypi.python.org/pypi/raytraverse

.. image:: https://readthedocs.org/projects/raytraverse/badge/?version=latest
        :target: https://raytraverse.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

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

    pip install clipt

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

