#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for raytraverse.scene"""
import os
import shutil

import numpy as np
import pytest

from raytraverse import renderer
from raytraverse.lightfield import LightResult


@pytest.fixture(scope="module")
def tmpdir(tmp_path_factory):
    data = str(tmp_path_factory.mktemp("data"))
    subf = 'tests/example/'
    shutil.copytree(subf, data + '/test')
    cpath = os.getcwd()
    path = data + '/test'
    # uncomment to use actual (to debug results)
    # path = cpath + '/' + subf
    os.chdir(path)
    yield path
    os.chdir(cpath)


def test_example(tmpdir):
    from raytraverse.example import main, out, scene_files, zone, epw, output
    print(renderer.Rcontrib.__dict__)
    main()
    print(renderer.Rcontrib.__dict__)
    lr1 = LightResult("check.npz")
    lr2 = LightResult(output)
    data, axes, names = lr1.pull("sky", "metric", findices=[[0], [3]])
    data2, axes, names = lr2.pull("sky", "metric", findices=[[0], [3]])
    assert np.allclose(data, data2, rtol=.3)
