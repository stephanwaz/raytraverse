===================
raytraverse (1.3.4)
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

Windows
~~~~~~~

Currently raytraverse is only compatible with macOS and linux operating systems. 
One way to use raytraverse on a Windows machine is with Docker. In adddition to the Docker
installation, this process will require about 1.5 GB of disk space.

1. Install Docker from: https://www.docker.com/products/docker-desktop/ 
   (click on "Windows") and then follow the installation instructions.
2. Open the newly installed Docker Desktop application (you do not need to sign in or create an account)
3. Create a new local folder and with a file named Dockerfile::

	 # syntax=docker/dockerfile:1
	 FROM python:3.9
	 WORKDIR /working
	 SHELL ["/bin/bash", "-c"]
	 RUN pip3 install raytraverse
	 
	 CMD raytraverse --help

4. in a command prompt navigate to this folder and::

	docker build . --tag raytraverse

5. To use raytraverse, navigate to a local folder that contains all necessary 
   files (radiance scene files, sky data, etc.).
6. Now, in this folder::

	docker run -it --name rayt --mount type=bind,source="$(pwd)",target=/working raytraverse /bin/bash

7. You now have a linux/bash command prompt in an environment with raytraverse 
   installed. The currrent directory will be named "working" within the linux environment 
   and is a shared resource with the host (changes on the host side are immediately seen in the container and vice
   versa). When you are finished, exit the linux shell ("exit"), then in the (now) windows command prompt::
   
	docker rm rayt

8. for ease of use, you can put these to lines in a .bat file somewhere in your execution PATH, 
   just make sure that docker desktop is running before calling::

	docker run -it --name rayt --mount type=bind,source="$(pwd)",target=/working raytraverse /bin/bash
	docker rm rayt
	
9. to update raytraverse, just repeat step 4 in a directory with the Dockerfile in step 3.
10. see the Docker settings for information about resource allocation to the docker container

Usage
-----
raytraverse includes a complete command line interface with all commands
nested under the `raytraverse` parent command enter::

    raytraverse --help

raytraverse also exposes an object oriented API written primarily in python.
calls to Radiance are made through Renderer objects that wrap the radiance
c source code in c++ classes, which are made available in python with pybind11.
see craytraverse (https://pypi.org/project/craytraverse/).

For complete documentation of the API and the command line interface either
use the Documentation link included above or::

    pip install -r docs/requirements.txt
    make docs

to generate local documentation.


