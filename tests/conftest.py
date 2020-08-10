#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import _pytest.skipping


def pytest_addoption(parser):
    parser.addoption(
        "--slowtest",
        action="store_true",
        default=False, help="disable skip marks")


@pytest.hookimpl(tryfirst=True)
def pytest_cmdline_preparse(config, args):
    if "--slowtest" not in args:
        return

    def no_skip(*args, **kwargs):
        return

    _pytest.skipping.skip = no_skip
