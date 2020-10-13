#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for raytraverse.scene"""
import os
import shutil

import pytest
from click.testing import CliRunner
import numpy as np
import clasp.click_ext as clk

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


@pytest.mark.slow
def test_cli(tmpdir, capfd, runner):
    with pytest.raises(SystemExit) as exc_info:
        with capfd.disabled():
            cli.main.main(args=["-c", "run.cfg", "demo", "sky", "sunrun"])
    assert exc_info.value.args[0] == 0
    pt = runner.invoke(cli.main, "-c run.cfg demo integrate")
    hdr = runner.invoke(hdrcli.img_cr, "'demo_view*.hdr'")
    hdr = np.fromstring(hdr.output, sep=' ').reshape(-1, 5)[:, 1:3]
    pt = np.fromstring(pt.output, sep=' ')
    corr = corr_calc(hdr[:, 0], pt)
    assert corr[0] > .95
