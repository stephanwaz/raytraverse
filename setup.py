#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2020 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

# c++ compiling and pybind setup code copied from https://github.com/pybind/python_example.git

# License:

# Copyright (c) 2016 The Pybind Development Team, All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# You are under no obligation whatsoever to provide any bug fixes, patches, or
# upgrades to the features, functionality or performance of the source code
# ("Enhancements") to anyone; however, if you choose to make your Enhancements
# available either publicly, or directly to the author of this software, without
# imposing a separate written license agreement for such Enhancements, then you
# hereby grant the following license: a non-exclusive, royalty-free perpetual
# license to install, use, modify, prepare derivative works, incorporate into
# other computer software, distribute, and sublicense such enhancements or
# derivative works thereof, in binary and source code form.


"""The setup script."""
from setuptools import find_packages
from skbuild import setup

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['clasp', 'numpy', 'scipy', 'pywavelets', 'matplotlib',
                'skyfield', 'clipt', 'pybind11']

setup_requirements = ["setuptools", "wheel", "scikit-build", "cmake", "ninja"]

test_requirements = ['pytest', 'hdrstats', 'pytest-cov']

# TODO: add necessary radiance .cal files (and make sure source files are
# included in in dist before creating pypi.
data_files = []
package_data = {"raytraverse": ["*.pl", "*.bsp", "cal/*.cal"]}

exclude_package_data = {"": ['*.rst', 'LICENSE', 'MAkefile', '*.ini', '*.in',
                             '*.txt', '*.sh', '*.toml', 'MANIFEST', 'tests',
                             'docs', 'src' ]}

setup_dict = dict(
    author="Stephen Wasilewski",
    author_email='stephanwaz@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        ],
    description="variance based adaptive sampling for CBDM",
    python_requires=">=3.6",
    entry_points={
        'console_scripts': [
            'raytraverse=raytraverse.cli:main',
            'genskyvec_sc=raytraverse.gsv:main'
            ],
        },
    cmake_minimum_required_version="3.15",
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
    version='0.1.1',
    zip_safe=False,
    )

setup(**setup_dict)
