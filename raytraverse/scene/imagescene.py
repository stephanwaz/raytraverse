# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
from raytraverse.formatter import Formatter
from raytraverse.scene.basescene import BaseScene


class ImageScene(BaseScene):
    """scene for image sampling

    Parameters
    ----------
    outdir: str
        path to store scene info and output files
    scene: str, optional
        image file (hdr format -vta projection)
    """

    def __init__(self, outdir, scene=None, formatter=Formatter, reload=True,
                 log=False):
        super().__init__(outdir, scene=scene, formatter=formatter, frozen=True,
                         reload=reload, log=log)
        self._logf = None

