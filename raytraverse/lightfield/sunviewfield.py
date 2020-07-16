# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

import os
import pickle
import itertools

import numpy as np
from scipy.cluster.hierarchy import fclusterdata

from raytraverse import translate, plot
from raytraverse.helpers import ArrayDict
from raytraverse.lightfield.lightfield import LightField
from raytraverse.mapper import SunMapper


class SunViewField(LightField):
    """container for sun view data

    Parameters
    ----------
    scene: raytraverse.scene.Scene
        scene class containing geometry, location and analysis plane
    suns: raytraverse.sunsetter.SunSetter
        prefix of data files to integrate
    rebuild: bool, optional
        build kd-tree even if one exists
    """
    nullvlo = np.zeros((1, 5))

    def __init__(self, scene, suns, rebuild=False):
        #: raytraverse.sunsetter.SunSetter
        self.suns = suns
        #: raytraverse.sunmapper.SunMapper
        self.sunmap = suns.map
        super().__init__(scene, rebuild=rebuild, prefix='sunview')

    @property
    def vlo(self):
        """sunview data indexed by (point, sun)

        key (i, j) val: np.array (N, 5) x, y, z, lum, omega

        :type: raytraverse.translate.ArrayDict
        """
        return self._vlo

    @property
    def raster(self):
        """sunview data indexed by (point, sun)

        key (i, j) val: np.array (N, M, 3) individual rays for interpolating
        to pixels

        :type: dict
        """
        return self._raster

    @property
    def scene(self):
        """scene information

        :getter: Returns this integrator's scene
        :setter: Set this integrator's scene
        :type: raytraverse.scene.Scene
        """
        return self._scene

    @scene.setter
    def scene(self, scene):
        """Set this integrator's scene and load samples"""
        self._scene = scene
        dfile = f'{self.scene.outdir}/{self.prefix}_result.pickle'
        if not os.path.isfile(dfile):
            raise FileNotFoundError("No results files found, have you run"
                                    f" a Sampler of type {self.prefix} for"
                                    f" scene {self.scene.outdir}?")
        kdfile = f'{self.scene.outdir}/{self.prefix}_kd_data.pickle'
        if os.path.isfile(kdfile) and not self.rebuild:
            f = open(kdfile, 'rb')
            self._vlo = pickle.load(f)
            self._raster = pickle.load(f)
            f.close()
        else:
            f = open(dfile, 'rb')
            vls = [pickle.load(f) for i in range(3)]
            f.close()
            self._vlo, self._raster = self._build_clusters(*vls)
            f = open(kdfile, 'wb')
            pickle.dump(self.vlo, f, protocol=4)
            pickle.dump(self.raster, f, protocol=4)
            f.close()

    def items(self):
        return itertools.product(super().items(), range(self.suns.shape[0]))

    def _cluster(self, vecs, lums, shape, suni):
        """group adjacent rays within sampling tolerance"""
        scalefac = ((self.sunmap.viewangle/2*np.pi/180)**2)
        omega0 = scalefac*np.pi/np.prod(shape)
        maxr = 2*np.sqrt(2)/shape[0]
        xy = translate.uv2xy(vecs)*shape[0]/(shape[0] - 1)
        cluster = fclusterdata(vecs, maxr, 'distance')
        xyz = self.sunmap.uv2xyz(vecs, i=suni)
        vlo = []
        raster = []
        for cidx in range(np.max(cluster)):
            grp = cluster == cidx + 1
            ptxy = xy[grp]
            ptxyz = xyz[grp]
            lgrp = lums[grp]
            uv = vecs[grp]
            raster.append(uv)
            vlo.append([*np.mean(ptxyz, 0), np.mean(lgrp), omega0*len(ptxy)])
        return np.array(vlo), raster

    def _build_clusters(self, vecs, lums, shape):
        """loop through points/suns and group adjacent rays"""
        vlo = ArrayDict({(-1, -1): self.nullvlo})
        raster = {(-1, -1): None}
        iterator = itertools.product(range(np.product(self.scene.ptshape)),
                                     range(self.suns.suns.shape[0]))
        for i, j in iterator:
            if len(vecs[i][j]) > 0:
                v, r = self._cluster(vecs[i][j], lums[i][j], shape, j)
                vlo[(i, j)] = v
                raster[(i, j)] = r
        return vlo, raster

    def draw_sun(self, psi, sun, vm, res):
        """

        Parameters
        ----------
        psi
        sun
        vm: raytraverse.mapper.ViewMapper
        res

        Returns
        -------

        """
        sunpix = None
        sunvals = None
        if psi in self.vlo.keys():
            vlos = self.vlo[psi]
            sm = SunMapper(sun[0:3])
            rys = self.raster[psi]
            sundict = {}
            for vlo, r in zip(vlos, rys):
                v = vlo[0:3]
                if vm.radians(v) <= vm.viewangle/2:
                    rxyz = sm.uv2xyz(r)
                    lm = vlo[3]
                    omega = vlo[4]
                    ppix = vm.ray2pixel(rxyz, res)
                    px, i, cnt = np.unique(np.core.records.fromarrays(ppix.T),
                                           return_index=True,
                                           return_counts=True)
                    omegap = vm.pixel2omega(ppix[i] + .5, res)*4/np.pi
                    omegasp = omega/r.shape[0]
                    clum = sun[-1] * lm * cnt * omegasp / omegap
                    for p, cl in zip(px, clum):
                        pt = tuple(p)
                        if pt in sundict:
                            sundict[pt] += cl
                        else:
                            sundict[pt] = cl
            sunpix = np.array(list(sundict.keys()))
            sunvals = np.array(list(sundict.values()))
        return sunpix, sunvals

    def direct_view(self, res=3):
        """create a summary image of all sun discs from each of vpts"""
        vpts = self.scene.pts()
        ssq = int(np.ceil(np.sqrt(self.suns.suns.shape[0])))
        square = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
        block = []
        for i in np.arange(self.suns.suns.shape[0], ssq**2):
            sxy = np.unravel_index(i, (ssq, ssq))
            sxy = np.array((sxy[1], sxy[0]))
            block.append(((1, 1, 1), square + sxy))
        for i, pt in enumerate(vpts):
            (lums, fig, ax, norm, lev) = plot.mk_img_setup([0, 1],
                                         figsize=[ssq*3, ssq*3], ext=(0, ssq))
            cmap = plot.colormap('viridis', norm)
            plot.plot_patches(ax, block)
            for j in range(self.suns.suns.shape[0]):
                if (i, j) in self.raster.keys():
                    sxy = np.unravel_index(j, (ssq, ssq))
                    sxy = np.array((sxy[1], sxy[0]))
                    lums = cmap.to_rgba(self.vlo[(i, j)][:, 3])
                    for r, lm in zip(self.raster[(i, j)], lums):
                        xy = (translate.uv2xy(r)+1)/2 + sxy
                        ax.plot(xy[:, 0], xy[:, 1], 'o', ms=1, color=lm)
            ax.set_facecolor((0, 0, 0))
            outf = f"{self.outfile(i)}.png"
            plot.save_img(fig, ax, outf, title=pt)
