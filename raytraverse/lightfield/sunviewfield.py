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
from clipt import mplt


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
        vlo = ArrayDict({(-1, -1): np.zeros((1, 5))})
        raster = {(-1, -1): None}
        iterator = itertools.product(range(np.product(self.scene.ptshape)),
                                     range(self.suns.suns.shape[0]))
        for i, j in iterator:
            if len(vecs[i][j]) > 0:
                v, r = self._cluster(vecs[i][j], lums[i][j], shape, j)
                vlo[(i, j)] = v
                raster[(i, j)] = r
        return vlo, raster

    def add_to_img(self, img, pi, sun, vm):
        """
        Parameters
        ----------
        img
        pi
        sun
        vm: raytraverse.mapper.ViewMapper

        Returns
        -------

        """
        res = img.shape[0]
        if pi not in self.vlo.keys():
            return None
        vlos = self.vlo[pi]
        sm = SunMapper(sun[0:3])
        rys = self.raster[pi]
        for vlo, r in zip(vlos, rys):
            v = vlo[0:3]
            if vm.radians(v) <= vm.viewangle/2:
                rxyz = sm.uv2xyz(r)
                lm = vlo[3]
                omega = vlo[4]
                # assign sample rays to pixels
                ppix = vm.ray2pixel(rxyz, res)
                rec = np.core.records.fromarrays(ppix.T)
                px, i, cnt = np.unique(rec, return_index=True,
                                       return_counts=True)
                omegap = vm.pixel2omega(ppix[i] + .5, res)
                omegasp = omega / r.shape[0]
                # xs = []
                # ys = []
                # xy = vm.xyz2xy(rxyz)
                # for j in i:
                #     xs.append(xy[rec == rec[j], 0])
                #     ys.append(xy[rec == rec[j], 1])
                # mplt.quick_scatter(xs, ys, ms=4, lw=0)
                np.set_printoptions(3, suppress=True)
                cnt = cnt.astype(float)
                # smudge (hack to ensure equal energy and max luminance)
                ocnt = cnt - (omegap/omegasp)
                smdg = np.sum(ocnt[ocnt > 0])
                cnt[ocnt > 0] = omegap[ocnt > 0]/omegasp
                # average to redistribute
                redist = smdg/np.sum(ocnt < 0)
                # redistribute over pixels with "room" (this could still
                # overshoot if too many pixels are close to threshold, but
                # maybe mathematically impossible?
                cnt[ocnt < -redist] += smdg/np.sum(ocnt < -redist)
                # apply average luminanace over each pixel
                clum = sun[-1] * lm * cnt * omegasp / omegap
                for p, cl in zip(px, clum):
                    img[tuple(p)] += cl

    def get_illum(self, vm, pis, coefs, scale=179):
        ct = np.maximum(np.einsum("ki,ji->jk", vm.dxyz, coefs[:, 0:3]), 0)
        rpt = int(len(pis)/ct.shape[0])
        ctheta = np.broadcast_to(ct[None, ...], (rpt,) + ct.shape)
        ctheta = ctheta.reshape(-1, ct.shape[1])
        sun = np.broadcast_to(coefs[None, :, -1], (rpt, coefs.shape[0])).ravel()
        hassun = np.array([pi in self.vlo.keys() for pi in pis])
        pis = np.array(pis)
        vlo = self.vlo[pis[hassun]]
        illum = np.zeros((len(pis), len(vm.dxyz)))
        illum[hassun] = np.einsum("i,i,ij,i,->ij", vlo[:, 3], vlo[:, 4],
                                  ctheta[hassun], sun[hassun], scale)
        illum = illum.reshape((-1, coefs.shape[0], vm.dxyz.shape[0]))
        return np.swapaxes(illum, 1, 2)

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
