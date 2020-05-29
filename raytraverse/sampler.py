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
        final directional resolution given as log2(res)
    skres: int
        side of square sky resolution (must be even)
    sunsperpatch: int
        maximum number of suns per sky patch to sample
    t0: float
        in range 0-1, fraction of uniform random samples taken at first step
    t1: float:
        in range 0-t0, fraction of uniform random samples taken at final step
    minrate: float:
        in range 0-1, fraction of samples at final step (this is not the total
        sampling rate, which depends on the number of levels).
    ipres: int:
        minimum position resolution (across maximum length of area)
    idres: int:
        initial direction resolution (as log2(res))
    """

    def __init__(self, scene, ptres=1.0, dndepth=9, skres=20, sunsperpatch=4,
                 t0=.1, t1=.01, minrate=.05, idres=4, ipres=4):
        self.scene = scene
        #: int: minimum position resolution (across maximum length of area)
        self.ipres = ipres
        #: int: initial direction resolution (as log2(res))
        self.idres = idres
        self.levels = (ptres, dndepth, skres)
        #: int: maximum number of suns per skypatch
        self.sunsperpatch = sunsperpatch
        #: float: fraction of uniform random samples taken at first step
        self.t0 = t0
        #: float: fraction of uniform random samples taken at final step
        self.t1 = t1
        #: float: fraction of samples at final step
        self.minrate = minrate
        #: true: set to True after call to this.mkpmap
        self.skypmap = False

    @property
    def levels(self):
        """sampling scheme

        :getter: Returns the sampling scheme
        :setter: Set the sampling scheme from (ptres, dndepth, skres)
        :type: np.array
        """
        return self._levels

    @levels.setter
    def levels(self, res):
        """calculate sampling scheme"""
        ptres, dndepth, skres = res
        bbox = self.scene.area.bbox[:, 0:2]/ptres
        size = (bbox[1] - bbox[0])
        uvlevels = np.floor(np.log2(np.max(size)/self.ipres)).astype(int)
        uvpow = 2**uvlevels
        uvsize = np.ceil(size/uvpow)*uvpow
        plevels = np.stack([uvsize/2**(uvlevels-i)
                            for i in range(uvlevels+1)])
        dlevels = np.array([(2**(i+1), 2**i)
                            for i in range(self.idres, dndepth+1, 1)])
        plevels = np.pad(plevels, [(0, dlevels.shape[0] - plevels.shape[0]),
                                   (0, 0)], mode='edge')
        slevels = np.full(dlevels.shape, skres)
        self._levels = np.hstack((plevels, dlevels, slevels)).astype(int)

    @property
    def scene(self):
        """scene information

        :getter: Returns this sampler's scene
        :setter: Set this sampler's scene and create sky octree
        :type: raytraverse.scene.Scene
        """
        return self._scene

    @scene.setter
    def scene(self, scene):
        """calculate sampling scheme"""
        self._scene = scene
        skydef = ("void light skyglow 0 0 3 1 1 1 skyglow source sky 0 0 4"
                  " 0 0 1 180")
        f = open(f'{self.scene.outdir}/sky.oct', 'wb')
        cst.pipeline([f'oconv -i {self.scene.outdir}/scene.oct -'], inp=skydef,
                     outfile=f, close=True)

    def mkpmap(self, apo, nproc=12, overwrite=False, nphotons=1e8,
               executable='mkpmap_dc', opts=''):
        apos = '-apo ' + ' -apo '.join(apo.split())
        if overwrite:
            force = '-fo+'
        else:
            force = '-fo-'
        fdr = self.scene.outdir
        cmd = (f'{executable} {opts} -n {nproc} {force} {apos} -apC "'
               f'"{fdr}/sky.gpm {nphotons} {fdr}/sky.oct')
        r, err = cst.pipeline([cmd], caperr=True)
        if b'fatal' in err:
            raise ChildProcessError(err.decode(cst.encoding))
        self.skypmap = True
