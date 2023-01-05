#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for raytraverse.scene"""
import os
import shutil

import pytest
from raytraverse.scene import Scene
import numpy as np


@pytest.fixture(scope="module")
def tmpdir(tmp_path_factory):
    data = str(tmp_path_factory.mktemp("data"))
    shutil.copytree('tests/scene/', data + '/test')
    cpath = os.getcwd()
    # use temp
    path = data + '/test'
    # uncomment to use actual (to debug results)
    # path = cpath + '/tests/scene'
    os.chdir(path)
    yield path
    os.chdir(cpath)


def test_reflection_search(tmpdir):
    scn = Scene("test", "north_glass.rad office_n.rad")
    vecs = np.array((0, -1, 1.2))[None]
    assert np.allclose(scn.reflection_search(vecs).ravel(), (0, -1, 0))
