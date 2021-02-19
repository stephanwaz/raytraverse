#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

"""The setup script."""
from setuptools import find_packages, setup
import sys
from skbuild import setup as sksetup

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['clasp', 'numpy', 'scipy', 'matplotlib',
                'skyfield', 'clipt', 'pybind11', 'shapely']

setup_requirements = ["setuptools", "wheel", "scikit-build", "cmake", "ninja"]

test_requirements = ['pytest', 'pytest-cov']

data_files = []
package_data = {"raytraverse": ["*.bsp", "cal/*.cal"]}

setup_dict = dict(
    author="Stephen Wasilewski",
    author_email='stephanwaz@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8'
        ],
    description="adaptive sampling for CBDM",
    python_requires=">=3.6",
    entry_points={
        'console_scripts': ['raytraverse=raytraverse.cli:main'],
        },
    install_requires=requirements,
    license="Mozilla Public License 2.0 (MPL 2.0)",
    long_description=readme + '\n\n' + history,
    include_package_data=False,
    keywords='raytraverse',
    name='raytraverse',
    packages=find_packages(),
    data_files=data_files,
    package_data=package_data,
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/stephanwaz/raytraverse',
    version='1.1.2',
    zip_safe=False,
    )

orig_argv = [i for i in sys.argv]
# scikit-build does not seem to package source correctly, revert to setuptools
morecmds = True
if 'sdist' in sys.argv:
    sdistargs = ['sdist']
    sstart = orig_argv.index('sdist')
    send = sstart + 1
    for i in orig_argv[send:]:
        if i[0] == "-":
            sdistargs.append(i)
            send += 1
        else:
            break
    sys.argv = [sys.argv[0]] + sdistargs
    setup(**setup_dict)
    sys.argv = orig_argv[:sstart] + orig_argv[send:]
    morecmds = len(sys.argv) > 1
# run the remaining commands with scikit-build
if morecmds:
    setup_dict["cmake_minimum_required_version"] = "3.15"
    sksetup(**setup_dict)

# install executables to bin/ with develop install
if 'develop' in sys.argv:
    from distutils import dir_util
    import glob
    import os
    try:
        buildbin = glob.glob("_skbuild/*/cmake-install/bin")[0]
    except IndexError:
        print("Warning no executables built", file=sys.stderr)
    else:
        dest = os.path.dirname(sys.executable)
        dir_util.copy_tree(buildbin, dest)
