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

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['craytraverse>=0.1.5', 'clasp', 'numpy', 'scipy', 'matplotlib', 'tqdm',
                'skyfield', 'shapely', 'sklearn']

setup_requirements = ["setuptools", "wheel"]

test_requirements = ['pytest', 'pytest-cov']

data_files = []
package_data = {"raytraverse": ["*.bsp"]}

setup(
    author="Stephen Wasilewski",
    author_email='stephanwaz@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        ],
    description="adaptive sampling for CBDM",
    python_requires=">=3.7",
    entry_points={
        'console_scripts': ['raytraverse=raytraverse.cli:main',
                            'raytu=raytraverse.raytu:main'],
        },
    install_requires=requirements,
    license="Mozilla Public License 2.0 (MPL 2.0)",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='raytraverse',
    name='raytraverse',
    packages=find_packages(),
    data_files=data_files,
    package_data=package_data,
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/stephanwaz/raytraverse',
    version='1.3.7',
    zip_safe=False,
    )
