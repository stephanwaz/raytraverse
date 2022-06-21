# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import os
import re
import sys

import numpy as np
from clasp.script_tools import pipeline

from raytraverse.mapper import ViewMapper
from raytraverse.renderer import SpRenderer
from raytraverse.scene.basescene import BaseScene
from raytraverse.formatter import RadianceFormatter


class Scene(BaseScene):
    """container for radiance scene description

    WARNING!! if scene parameter contains and instance primitive, sunsampler
    will throw a segmentation fault when it tries to change the source. As
    scene instanciation will make a frozen octree, it is better to feed complete
    scene description files, or an octree.

    Parameters
    ----------
    outdir: str
        path to store scene info and output files
    formatter: raytraverse.formatter.RadianceFormatter, optional
        intended renderer format
    """
    def __init__(self, outdir, scene=None, frozen=True,
                 formatter=RadianceFormatter, **kwargs):
        self._refl_scene = None
        super().__init__(outdir, scene=scene, frozen=frozen,
                         formatter=formatter, **kwargs)

    def reflection_search_scene(self):
        octf = f"{self.outdir}/scene_reflections.oct"
        if os.path.isfile(octf):
            self._refl_scene = octf
        elif self._refl_scene is None:
            skyf = f"{self.outdir}/reflections_sky.rad"
            f = open(skyf, 'w')
            f.write(self.formatter.get_skydef((1, 1, 1), ground=False))
            f.close()
            files, frozen = self.formatter.get_scene(self.scene)
            if frozen:
                print(f"Warning, scene made from frozen octree or source scene "
                      f"files can no longer be located, reflection search will"
                      f"miss specular plastic", file=sys.stderr)
            pipeline([f"oconv -w {files} {skyf}"], outfile=octf, writemode='wb')
            self._refl_scene = octf
        return self._refl_scene

    def reflection_search(self, vecs, res=5):
        # plastic reflections do not work in a frozen octree, so need to try
        # and recompile.
        octf = self.reflection_search_scene()
        reflengine = SpRenderer("-ab 0 -w- -lr 1 -ss 0 -st .001 -otndM -h",
                                octf)
        vm = ViewMapper()
        side = 2**res
        uv = np.stack(np.unravel_index(np.arange(side*side*2),
                                       (2*side, side))).T/side
        uv += np.random.default_rng().random(uv.shape) * (.5 / side)
        xyz = vm.uv2xyz(uv)
        pvecs = np.concatenate(np.broadcast_arrays(vecs[:, None],
                                                   xyz[None, :]), 2)
        a = reflengine(pvecs.reshape(-1, 6))
        # count tabs to get level
        level = np.array([len(i) - len(i.lstrip())
                          for i in a.splitlines(False)])
        # normal, direction, modifier
        a = np.array(a.split()).reshape(-1, 7)
        mod = a[:, -1]
        a = a[:, 0:6].astype(float)
        sky = np.argwhere(mod == "skyglow").ravel()
        skyl = level[sky]
        leva, levs = np.broadcast_arrays(level, skyl[:, None])
        # filter indices before sky index in each row
        leva[np.arange(len(level)) <= sky[:, None]] = -3
        # filter indices not one level up from sky
        leva[level != (skyl[:, None] - 1)] = -2
        # this returns the first value, our candidate reflection
        candidate = np.argmax(leva, 1)
        # check if this is a reflection
        acos = np.einsum("ij,ij->i", a[candidate, 3:], a[sky, 3:])
        normals = a[candidate, 0:3][acos < .99]
        # find unique
        return np.array(list(set(zip(*normals.T))))

    def source_scene(self, srcfile, srcname):
        files, frozen = self.formatter.get_scene(self.scene)
        octf = f"{self.outdir}/scene_{srcname}.oct"
        pipeline([f"oconv -w {files} {srcfile}"], outfile=octf, writemode='wb')
        return octf
