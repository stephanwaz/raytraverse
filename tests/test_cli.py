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
    expected = [544.31400016,  3754.92591018,  5662.50520459,  6815.76788537,
                3915.48211025,  3677.63440588,   338.42069348,  3277.9985365,
                6805.48795167,  9633.50304282,  8297.96371376,  5057.53912396,
                13143.7072058,  13150.55951599, 14093.58852621, 15393.43980526,
                16319.97959988, 16729.46884444, 15845.99868256, 13428.96411586,
                2352.58732198]
    with pytest.raises(SystemExit) as exc_info:
        with capfd.disabled():
            cli.main.main(args=["-c", "run.cfg", "demo", "sky", "sunrun"])
    assert exc_info.value.args[0] == 0
    pt = runner.invoke(cli.main, "-c run.cfg demo integrate")
    pt = np.fromstring(pt.output, sep=' ')
    corr = corr_calc(expected, pt)
    assert corr[0] > .999
