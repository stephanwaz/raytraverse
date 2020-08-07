#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for raytraverse.scene"""
import os
import shutil

import pytest
from click.testing import CliRunner
import numpy as np

from raytraverse import cli
import hdrstats.cli as hdrcli
from hdrstats.hdrstats import corr_calc



@pytest.fixture()
def runner():
    return CliRunner(mix_stderr=False)

@pytest.fixture(scope="module")
def tmpdir(tmp_path_factory):
    data = str(tmp_path_factory.mktemp("data"))
    shutil.copytree('tests/workflow/', data + '/workflow')
    cpath = os.getcwd()
    os.chdir(data + '/workflow')
    yield data + '/workflow'
    os.chdir(cpath)


def test_cli(runner, tmpdir):
    result = runner.invoke(cli.main, "-c run.cfg demo sky sunrun")
    assert result.exit_code == 0
    result = runner.invoke(cli.main, "-c run.cfg demo integrate --no-illum")
    assert result.exit_code == 0
    hdr = runner.invoke(hdrcli.img_cr, "'demo_view*.hdr'")
    hdr = np.fromstring(hdr.output, sep=' ').reshape(-1, 5)[:, 1:3]
    pt = runner.invoke(cli.main, "-c run.cfg demo integrate --debug --no-hdr")
    pt = np.fromstring(pt.output, sep=' ').reshape(-1, 6)[:, 3:5]
    corr = corr_calc(hdr[:, 0], pt[:, 0])
    print(corr)
    assert corr[0] > .95
    corr = corr_calc(hdr[:, 1], pt[:, 1])
    print(corr)
    assert corr[0] > .95


