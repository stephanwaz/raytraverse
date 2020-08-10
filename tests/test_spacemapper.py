#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for raytraverse.scene"""
import os
import shutil

import pytest
from raytraverse.mapper import SpaceMapper
import numpy as np

# from clipt import mplt


@pytest.fixture(scope="module")
def tmpdir(tmp_path_factory):
    data = str(tmp_path_factory.mktemp("data"))
    shutil.copytree('tests/test/', data + '/test')
    cpath = os.getcwd()
    os.chdir(data + '/test')
    yield data + '/test'
    os.chdir(cpath)


def test_spacemapper(tmpdir):
    sm = SpaceMapper('plane.rad')
    assert np.all(np.equal(sm.pt2uv(sm.bbox), [[0, 0], [1, 1]]))
    sm2 = SpaceMapper('plane.rad', rotation=54)
    assert np.allclose(sm2.uv2pt(sm2._path[0].vertices), sm.uv2pt(sm._path[0].vertices))


