# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import numpy as np

from raytraverse.mapper import SpaceMapperPt
from raytraverse.scene.basescene import BaseScene
from raytraverse.formatter import ImageFormatter


class ImageScene(BaseScene):
    """scene for image sampling

    Parameters
    ----------
    outdir: str
        path to store scene info and output files
    scene: str, optional
        image file (hdr format -vta projection)
    """
    scene_ext = "hdr"

    def __init__(self, outdir, scene=None, reload=True):
        super().__init__(outdir, scene=scene, viewdir=(0, 1, 0), viewangle=180,
                         frozen=True, formatter=ImageFormatter, reload=reload)
        self.area = SpaceMapperPt(np.array((0, 0, 0)))
