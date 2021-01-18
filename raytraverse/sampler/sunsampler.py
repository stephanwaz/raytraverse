# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import os

import numpy as np

from raytraverse import translate, draw
from raytraverse.lightfield import SCBinField
from raytraverse.sampler.sampler import Sampler


class SunSampler(Sampler):
    """sample contributions from direct suns.

    Parameters
    ----------
    scene: raytraverse.scene.Scene
        scene class containing geometry, location and analysis plane
    sun: np.array
        shape 3, sun position
    sunbin: int
        sun bin
    ropts: str, optional
        arguments for engine
    speclevel: int, optional
        at this sampling level, pdf is made from brightness of sky sampling
        rather than progressive variance to look for fine scale specular
        highlights, this should be atleast 1 level from the end and the
        resolution of this level should be smaller than the size of the source
    keepamb: bool, optional
        whether to keep ambient files after run, if kept, a successive call
        will load these ambient files, so care must be taken to not change
        any parameters
    ambcache: bool, optional
        whether the rcopts indicate that the calculation will use ambient
        caching (and thus should write an -af file argument to the engine)
    """

    def __init__(self, scene, sun, sunbin, speclevel=9, fdres=10,
                 ropts='-ab 6 -ad 3000 -as 1500 -st 0 -ss 16 -aa .1',
                 keepamb=False, ambcache=True, slimit=0.01, **kwargs):
        self.slimit = slimit
        self.specguide = None
        # update ambient file and args before init
        self._keepamb = keepamb and ambcache
        if ambcache:
            self.ambfile = f"{scene.outdir}/sun_{sunbin:04d}.amb"
        else:
            self.ambfile = None
        engine_args = scene.formatter.get_standard_args(ropts, self.ambfile)
        super().__init__(scene, stype=f"sun_{sunbin:04d}", fdres=fdres,
                         engine_args=engine_args, **kwargs)
        # update parameters post init
        # normalize accuracy for sun source
        self.accuracy = self.accuracy * (1 - np.cos(.533*np.pi/360))
        #: int: index of level at which brightness sampling occurs
        self.specidx = speclevel - self.idres
        #: np.array: sun position x,y,z
        self.sunpos = sun

        # load new source
        srcdef = f'{scene.outdir}/tmp_srcdef_{sunbin}.rad'
        f = open(srcdef, 'w')
        f.write(scene.formatter.get_sundef(sun, (1, 1, 1)))
        f.close()
        self.engine.load_source(srcdef)
        os.remove(srcdef)

    # move to skypoint
    def pdf_from_sky(self, rebuild=False, zero=True,
                     filterpts=False):
        # add some tolerance for suns near edge of bins:
        uv = translate.xyz2uv(self.sunpos[None, :], flipu=False)
        tol = .125/self.scene.skyres
        uvi = np.linspace(-tol, tol, 3)
        uvs = np.stack(np.meshgrid(uvi, uvi)).reshape(2, 9).T + uv
        sbin = np.unique(translate.uv2bin(uvs, self.scene.skyres)).astype(int)
        self.sbin = sbin[sbin <= self.scene.skyres**2]

        skyfield = SCBinField(self.scene, log=False)
        ishape = np.concatenate((self.area.ptshape,
                                 self.levels[self.specidx-2]))
        fi = f"{self.scene.outdir}/sunpdfidxs.npz"
        if os.path.isfile(fi) and not rebuild:
            f = np.load(fi)
            idxs = f['arr_0']
        else:
            shp = self.levels[self.specidx-2]
            si = np.stack(np.unravel_index(np.arange(np.product(shp)), shp))
            uv = (si.T + .5)/shp[1]
            grid = skyfield.scene.view.uv2xyz(uv)
            idxs, errs = skyfield.query_all_pts(grid)
            strides = np.array(skyfield.lum.index_strides()[:-1])[:, None, None]
            idxs = np.reshape(np.atleast_3d(idxs) + strides, (-1, 1))
            np.savez(fi, idxs)
        column = skyfield.lum.full_array()
        lum = np.max(column[idxs, self.sbin], -1).reshape(ishape)
        if filterpts:
            haspeak = np.max(lum, (2, 3)) > self.srct
            lum = np.where(haspeak[..., None, None], 1.0, 0)
        elif zero:
            lum = np.where(lum > self.scene.maxspec, 0, lum)
        return lum

    def sample(self, vecf, vecs):
        """call rendering engine to sample sky contribution"""
        return super().sample(vecf, vecs).ravel()

    # @profile
    def draw(self, level):
        """draw samples based on detail calculated from weights
        detail is calculated across direction only as it is the most precise
        dimension

        Returns
        -------
        pdraws: np.array
            index array of flattened samples chosen to sample at next level
        """
        pdraws, pa = super().draw(level)
        s = 0
        if level == self.specidx and self.specguide is not None:
            shape = self.levels[level]
            p = translate.resample(self.specguide, shape)
            s = p.ravel()
            s[pdraws] = 0
            sdraws = draw.from_pdf(s, self.slimit, ub=1)
            pdraws = np.concatenate((pdraws, sdraws))
        return pdraws, pa + s

    def run_callback(self, vecfs):
        super().run_callback(vecfs)
        if not self._keepamb:
            try:
                os.remove(self.ambfile)
            except (IOError, TypeError):
                pass

    def run(self, point, vm, plotp=False, specguide=None, **kwargs):
        self.specguide = specguide
        super().run(point, vm, plotp, **kwargs)
