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
from memory_profiler import profile

class SingleSunSampler(Sampler):
    """sample contributions from direct suns.

    Parameters
    ----------
    scene: raytraverse.scene.Scene
        scene class containing geometry, location and analysis plane
    suns: raytraverse.sunsetter.SunSetter
        sun class containing sun locations.
    sidx: int
        sun index to sample
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

    def __init__(self, scene, suns, sidx, speclevel=9,
                 fdres=10, rcopts='-ab 6 -ad 3000 -as 1500 -st 0 -ss 16 -aa .1',
                 keepamb=False, ambcache=True, **kwargs):
        #: float: controls sampling limit in case of limited contribution
        self.slimit = suns.srct * .5
        self.srct = suns.srct
        # update ambient file and args before init
        self._keepamb = keepamb and ambcache
        if ambcache:
            ambfile = f"{scene.outdir}/sun_{sidx:04d}.amb"
        else:
            ambfile = None
        engine_args = scene.formatter.get_standard_args(rcopts, ambfile)
        super().__init__(scene, stype=f"sun_{sidx:04d}", fdres=fdres,
                         engine_args=engine_args, **kwargs)
        # update parameters post init
        # normalize accuracy for sun source
        self.accuracy = self.accuracy * (1 - np.cos(.533*np.pi/360))
        #: int: index of level at which brightness sampling occurs
        self.specidx = speclevel - self.idres
        self.sidx = sidx
        #: np.array: sun position x,y,z
        self.sunpos = suns.suns[sidx]

        # add some tolerance for suns near edge of bins:
        uv = translate.xyz2uv(self.sunpos[None, :], flipu=False)
        tol = .125/self.scene.skyres
        uvi = np.linspace(-tol, tol, 3)
        uvs = np.stack(np.meshgrid(uvi, uvi)).reshape(2, 9).T + uv
        sbin = np.unique(translate.uv2bin(uvs, self.scene.skyres)).astype(int)
        self.sbin = sbin[sbin <= self.scene.skyres**2]

        # explude points with low direct sun patch contribution
        shape = np.concatenate((self.area.ptshape, self.levels[0]))
        weights = self.pdf_from_sky(filterpts=True)
        self.weights = translate.resample(weights, shape)

        # load new source
        srcdef = f'{scene.outdir}/tmp_srcdef.rad'
        f = open(srcdef, 'w')
        f.write(suns.write_sun(sidx))
        f.close()
        self.engine.load_source(srcdef)
        os.remove(srcdef)

    # @profile
    def pdf_from_sky(self, rebuild=False, zero=True,
                     filterpts=False):
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
    def draw(self):
        """draw samples based on detail calculated from weights
        detail is calculated across direction only as it is the most precise
        dimension

        Returns
        -------
        pdraws: np.array
            index array of flattened samples chosen to sample at next level
        """
        pdraws = super().draw()
        if self.idx == self.specidx:
            shape = np.concatenate((self.area.ptshape,
                                    self.levels[self.idx]))
            weights = self.pdf_from_sky()
            p = translate.resample(weights, shape)
            s = p.ravel()
            s[pdraws] = 0
            if self.plotp:
                self._plot_p(p, suffix="_specidx.hdr")
            sdraws = draw.from_pdf(s, self.slimit, ub=1)
            pdraws = np.concatenate((pdraws, sdraws))
        return pdraws

    def run_callback(self):
        super().run_callback()
        if not self._keepamb:
            try:
                os.remove(f'{self.scene.outdir}/sun_{self.sidx:04d}.amb')
            except (IOError, TypeError):
                pass
