# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import os

import numpy as np

from raytraverse import translate, draw, renderer
from raytraverse.lightfield import SCBinField
from raytraverse.sampler.sampler import Sampler


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
                 fdres=10, accuracy=1,
                 rcopts='-ab 6 -ad 3000 -as 1500 -st 0 -ss 16 -aa .1',
                 keepamb=False, ambcache=True, **kwargs):
        #: float: controls sampling limit in case of limited contribution
        self.slimit = suns.srct * .5
        self.srct = suns.srct
        anorm = accuracy * (1 - np.cos(.533*np.pi/360))
        self.engine = renderer.Rtrace()
        engine_args = f"{rcopts} -oZ"
        srcdef = f'{scene.outdir}/tmp_srcdef.rad'
        f = open(srcdef, 'w')
        f.write(suns.write_sun(sidx))
        f.close()
        if self.engine.Engine == "rtrace":
            srcdefcomp = srcdef
        else:
            srcdefcomp = None
        # update ambient file and args before init
        self._keepamb = keepamb and ambcache
        if ambcache:
            engine_args += f" -af {scene.outdir}/sun_{sidx:04d}.amb"
        super().__init__(scene, stype=f"sun_{sidx:04d}", fdres=fdres,
                         accuracy=anorm, engine_args=engine_args,
                         srcdef=srcdefcomp, **kwargs)

        # update parameters post init
        #: int: index of level at which brightness sampling occurs
        self.specidx = speclevel - self.idres
        self.sidx = sidx
        #: np.array: sun position x,y,z
        self.sunpos = suns.suns[sidx]
        # add some tolerance for suns near edge of bins:
        uv = translate.xyz2uv(self.sunpos[None, :], flipu=False)
        uvi = np.linspace(-.016667, .016667, 3)
        uvs = np.stack(np.meshgrid(uvi, uvi)).reshape(2, 9).T + uv
        self.sbin = np.unique(translate.uv2bin(uvs,
                                               self.scene.skyres)).astype(int)

        # load new source
        self.engine.load_source(srcdef)
        os.remove(srcdef)

    def pdf_from_sky(self, skyfield, rebuild=False, zero=True,
                     filterpts=False):
        ishape = np.concatenate((self.scene.area.ptshape,
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
            haspeak = np.max(lum, (2, 3)) > self.slimit
            lum = lum * haspeak[..., None, None]
        if zero:
            lum = np.where(lum > self.scene.maxspec, 0, lum)
        return lum

    def sample(self, vecf):
        """call rendering engine to sample sky contribution

        Parameters
        ----------
        vecf: str
            path of file name with sample vectors
            shape (N, 6) vectors in binary float format

        Returns
        -------
        lum: np.array
            array of shape (N,) to update weights
        """
        return super().sample(vecf).ravel()

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
            skyfield = SCBinField(self.scene)
            shape = np.concatenate((self.scene.area.ptshape,
                                    self.levels[self.idx]))
            weights = self.pdf_from_sky(skyfield)
            p = translate.resample(weights, shape)
            bound = self._linear(self.idx, .5, .01)
            s = p.ravel()
            s[pdraws] = 0
            if self.plotp:
                self._plot_p(p, suffix="_specidx.hdr")
            sdraws = draw.from_pdf(s, self.slimit,
                                   lb=1 - bound, ub=1 + bound)
            pdraws = np.concatenate((pdraws, sdraws))
        return pdraws

    def run_callback(self):
        super().run_callback()
        if not self._keepamb:
            try:
                os.remove(f'{self.scene.outdir}/sun_{self.sidx:04d}.amb')
            except (IOError, TypeError):
                pass
