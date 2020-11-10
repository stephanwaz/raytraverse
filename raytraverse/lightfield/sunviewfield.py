# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

import os
import pickle
import itertools
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from raytraverse import translate, plot
from raytraverse.lightfield.lightfield import LightField


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
    @staticmethod
    def _to_pix(rxyz, atv, vm, res):
        if atv > 90:
            ppix = vm.ivm.ray2pixel(rxyz, res)
            ppix[:, 0] += res
        else:
            ppix = vm.ray2pixel(rxyz, res)
        rec = np.core.records.fromarrays(ppix.T)
        px, i, cnt = np.unique(rec, return_index=True,
                               return_counts=True)
        cnt = cnt.astype(float)
        omegap = vm.pixel2omega(ppix[i] + .5, res)
        return px, omegap, cnt

    @staticmethod
    def _smudge(cnt, omegap, omegasp):
        """hack to ensure equal energy and max luminance)"""
        ocnt = cnt - (omegap/omegasp)
        smdg = np.sum(ocnt[ocnt > 0])
        cnt[ocnt > 0] = omegap[ocnt > 0]/omegasp
        # average to redistribute
        redist = smdg/np.sum(ocnt < 0)
        # redistribute over pixels with "room" (this could still
        # overshoot if too many pixels are close to threshold, but
        # maybe mathematically impossible?
        cnt[ocnt < -redist] += smdg/np.sum(ocnt < -redist)

    def __init__(self, scene, suns, rebuild=False, rmraw=False,
                 prefix='sunview', blursun=1.0):
        #: raytraverse.sunsetter.SunSetter
        self.suns = suns
        #: raytraverse.sunmapper.SunMapper
        self.sunmap = suns.map
        self.blursun = blursun
        super().__init__(scene, rebuild=rebuild, prefix=prefix, rmraw=rmraw)

    def raw_files(self):
        """get list of files used to build field"""
        dfile = f'{self.scene.outdir}/{self.prefix}_vals.out'
        vfile = f'{self.scene.outdir}/{self.prefix}_vecs.out'
        final = f'{self.scene.outdir}/{self.prefix}_result.pickle'
        return [dfile, vfile, final]

    @property
    def raster(self):
        """individual pixels forming shape of the sun, stored as uv coorinates
        with basis viewmapper about sun direction and diameter. indexed like
        vec, lum, and omega
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
        kdfile = f'{self.scene.outdir}/{self.prefix}_kd_data.pickle'
        if os.path.isfile(kdfile) and not self.rebuild:
            f = open(kdfile, 'rb')
            self._vec = pickle.load(f)
            self._lum = pickle.load(f)
            self._omega = pickle.load(f)
            self._raster = pickle.load(f)
            f.close()
        else:
            dfile = f'{self.scene.outdir}/{self.prefix}_result.pickle'
            if not os.path.isfile(dfile):
                raise FileNotFoundError("No results files found, have you run"
                                        f" a Sampler of type {self.prefix} for"
                                        f" scene {self.scene.outdir}?")
            f = open(dfile, 'rb')
            vls = [pickle.load(f) for i in range(3)]
            f.close()
            (self._vec, self._lum,
             self._omega, self._raster) = self._build_suns(*vls)
            f = open(kdfile, 'wb')
            pickle.dump(self.vec, f, protocol=4)
            pickle.dump(self.lum, f, protocol=4)
            pickle.dump(self.omega, f, protocol=4)
            pickle.dump(self.raster, f, protocol=4)
            f.close()

    def items(self):
        return self.vec.keys()

    def ptitems(self, i):
        return [j for j in self.items() if j[0] == i]

    def _grp_by_sun(self, vecs, lums, shape, pti, suni):
        scalefac = ((self.sunmap.viewangle/2*np.pi/180)**2)
        omega0 = scalefac*np.pi/np.prod(shape)
        xyz = self.sunmap.uv2xyz(vecs, i=suni)
        v = np.mean(xyz, 0)
        lm = np.mean(lums)
        og = omega0*len(vecs)
        return pti, suni, v, lm, og, vecs

    def _build_suns(self, vecs, lums, shape):
        """loop through points/suns and group adjacent rays"""
        vec = {}
        lum = {}
        omega = {}
        raster = {}
        futures = []
        with ThreadPoolExecutor() as exc:
            items = itertools.product(super().items(),
                                      range(self.suns.suns.shape[0]))
            for i, j in items:
                if len(vecs[i][j]) > 0:
                    futures.append(exc.submit(self._grp_by_sun, vecs[i][j],
                                              lums[i][j], shape, i, j))
            for fu in as_completed(futures):
                i, j, v, lm, og, ras = fu.result()
                vec[(i, j)] = v
                lum[(i, j)] = lm
                omega[(i, j)] = og
                raster[(i, j)] = ras
        return vec, lum, omega, raster

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
        res = img.shape[1]
        if pi not in self.vec.keys():
            return None
        # use actual sun location
        # sm = SunMapper(sun[0:3])
        atv = vm.degrees(self.vec[pi])
        if atv <= vm.viewangle/2:
            # rxyz = sm.uv2xyz(r)
            rxyz = self.sunmap.uv2xyz(self.raster[pi], pi[1])
            px, omegap, cnt = self._to_pix(rxyz, atv, vm, res)
            omegasp = self.omega[pi] / self.raster[pi].shape[0]
            self._smudge(cnt, omegap, omegasp)
            # apply average luminanace over each pixel
            clum = sun[3] * self.lum[pi] * cnt * omegasp / omegap
            for p, cl in zip(px, clum):
                img[tuple(p)] += cl

    def get_ray(self, psi, vm, s):
        sun = np.asarray(s[0:3]).reshape(1, 3)
        if vm.in_view(sun, indices=False)[0] and psi in self.items():
            svlm = self.lum[psi]*s[3]
            svo = self.omega[psi]
            return s[0:3], svlm/self.blursun, svo*self.blursun
        else:
            return None

    def direct_view(self, res=2):
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
            (lums, fig, ax,
             norm, lev) = plot.mk_img_setup([0, 1], figsize=[ssq*res, ssq*res],
                                            ext=(0, ssq))
            cmap = plot.colormap('viridis', norm)
            plot.plot_patches(ax, block)
            for j in range(self.suns.suns.shape[0]):
                if (i, j) in self.raster.keys():
                    sxy = np.unravel_index(j, (ssq, ssq))
                    sxy = np.array((sxy[1], sxy[0]))
                    lums = cmap.to_rgba(self.lum[(i, j)])
                    xy = (translate.uv2xy(self.raster[(i, j)])+1)/2 + sxy
                    ax.plot(xy[:, 0], xy[:, 1], 'o', ms=2, color=lums)
            ax.set_facecolor((0, 0, 0))
            outf = f"{self.outfile(i)}.png"
            plot.save_img(fig, ax, outf, title=pt)
