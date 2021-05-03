#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for raytraverse.scene"""
import os
import shutil

import pytest
from click.testing import CliRunner
import numpy as np

from raytraverse import cli, translate
from raytraverse.sky import skycalc
from matplotlib.testing.compare import compare_images


@pytest.fixture()
def runner():
    return CliRunner(mix_stderr=False)


@pytest.fixture(scope="module")
def tmpdir(tmp_path_factory):
    data = str(tmp_path_factory.mktemp("data"))
    shutil.copytree('tests/cli/', data + '/test')
    cpath = os.getcwd()
    # use temp
    path = data + '/test'
    # uncomment to use actual (to debug results)
    # path = cpath + '/tests/cli'
    os.chdir(path)
    yield path
    os.chdir(cpath)
