#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2020 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['clasp', 'numpy', 'scipy', 'pywavelets', 'matplotlib',
                'skyfield']

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

data_files = []
package_data = {}

setup(
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
        ],
    },
    install_requires=requirements,
    license="Mozilla Public License 2.0 (MPL 2.0)",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='raytraverse',
    name='raytraverse',
    packages=find_packages(include=['raytraverse']),
    data_files=data_files,
    package_data=package_data,
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://gitlab.enterpriselab.ch/lightfields/raytraverse',
    version='0.1.0',
    zip_safe=True,
)
