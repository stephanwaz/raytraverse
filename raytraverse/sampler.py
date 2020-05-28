# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

import os
import re
import shutil

import numpy as np

from clasp import script_tools as cst
from clasp.click_callbacks import parse_file_list
from raytraverse import SpaceMapper, sunpos, translate


class Sampler(object):
    """holds scene information and sampling scheme

    Parameters
    ----------
    scene: raytraverse.scene.Scene
        scene class containing geometry, location and analysis plane
    ptres: float
        final spatial resolution in scene geometry units
    dndepth: int
        final directional resolution given as log_2
    skres: int
        side of square sky resolution (must be even)
    sudepth: int
        number of doublings of sky resolution to do for sun sampling
    t0: float
        in range 0-1, fraction of uniform random samples taken at first step
    t1: float:
        in range 0-t0, fraction of uniform random samples taken at final step
    minrate: float:
        in range 0-1, fraction of samples at final step (this is not the total
        sampling rate, which depends on the number of levels).
    """

    def __init__(self, scene, ptres=1.0, dndepth=9, skres=20, sudepth=2,
                 t0=.1, t1=.01, minrate=.05):
        #: raytraverse.scene.Scene: scene description
        self.scene = scene




