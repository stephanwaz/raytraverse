#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import os
import sys
import craytraverse

failures = os.path.dirname(__file__) + "/failures"


def pytest_addoption(parser):
    parser.addoption(
        "--slow",
        action="store_true",
        default=False, help="disable slow skip marks")


def pytest_configure(config):
    craytraverse.set_raypath()

    path_rt = [os.path.dirname(sys.executable)]
    try:
        path_env = os.environ["PATH"].split(os.pathsep)
    except KeyError:
        path_new = path_rt
    else:
        path_new = list(dict.fromkeys(path_rt + path_env))
    os.environ["PATH"] = os.pathsep.join(path_new)

    if os.path.isfile(failures):
        os.remove(failures)
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--slow"):
        # --slow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --slow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    # execute all other hooks to obtain the report object
    outcome = yield
    rep = outcome.get_result()
    # we only look at actual failing test calls, not setup/teardown
    if rep.when == "call" and rep.failed:
        mode = "a" if os.path.exists(failures) else "w"
        with open(failures, mode) as f:
            f.write(rep.nodeid + "\n")
