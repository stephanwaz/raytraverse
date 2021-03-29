#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for raytraverse.scene"""
import os
import shutil

import pytest
from click.testing import CliRunner
import numpy as np

from raytraverse import cli, translate
from raytraverse.sky import SolarBoundary, skycalc
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


def test_cli_suns(tmpdir, runner):
    loc = skycalc.get_loc_epw("geneva.epw")
    sb = SolarBoundary(loc)

    pt = runner.invoke(cli.main, "error suns --usepositions")
    assert "ValueError" in pt.stderr

    pt = runner.invoke(cli.main, "epw suns --printsuns -wea geneva.epw --plotdview")
    pt = np.fromstring(pt.output, sep=' ').reshape(-1, 3)
    assert np.alltrue(sb.in_solarbounds(pt, size=-15))

    pt = runner.invoke(cli.main, "reload suns -wea epw/suns.dat --usepositions --plotdview")
    print(pt.output)
    assert compare_images("reload_suns.png", "epw_suns.png", .01) is None

    pt = runner.invoke(cli.main, "epw suns -loc '{} {} {}'".format(*loc))
    pt = np.fromstring(pt.output, sep=' ').reshape(-1, 3)
    assert np.alltrue(sb.in_solarbounds(pt))

    pt = runner.invoke(cli.main, "epw suns -wea geneva.epw -usepositions -skyro 30")
    pt = np.fromstring(pt.output, sep=' ').reshape(-1, 3)
    aa = translate.xyz2aa(pt)
    aa[:, 1] -= 30
    xyz = translate.aa2xyz(aa)
    assert np.alltrue(sb.in_solarbounds(xyz))

    pt = runner.invoke(cli.main, "epw suns -loc '0 175 180' -epw geneva.epw")
    pt = np.fromstring(pt.output, sep=' ').reshape(-1, 3)
    sb = SolarBoundary((0, 175, 180))
    assert np.alltrue(sb.in_solarbounds(pt))
