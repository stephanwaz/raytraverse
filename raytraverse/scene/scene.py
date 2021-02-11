# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
from raytraverse.scene.basescene import BaseScene
from raytraverse.formatter import RadianceFormatter


class Scene(BaseScene):
    """container for radiance scene description

    Parameters
    ----------
    outdir: str
        path to store scene info and output files
    formatter: raytraverse.formatter.RadianceFormatter, optional
        intended renderer format
    """
    def __init__(self, outdir, scene=None, frozen=True,
                 formatter=RadianceFormatter, **kwargs):
        super().__init__(outdir, scene=scene, frozen=frozen,
                         formatter=formatter, **kwargs)


