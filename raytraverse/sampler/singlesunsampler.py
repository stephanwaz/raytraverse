# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import os

import numpy as np

from raytraverse import translate, quickplot, draw
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
    """

    def __init__(self, scene, suns, sidx, sb=None, speclevel=9,
                 fdres=10, accuracy=1, keepamb=False, **kwargs):
        #: np.array: sun position x,y,z
        self.sunpos = suns.suns[sidx]
        #: int: sun index of sunpos from associated SunSetter (for naming)
        self.sidx = sidx
        if sb is None:
            sb = translate.uv2bin(translate.xyz2uv(self.sunpos[None, :],
                                                   flipu=False),
                                  scene.skyres).astype(int)[0]
        self.sbin = sb
        #: float: controls sampling limit in case of limited contribution
        self.slimit = suns.srct * .1
        self.srct = suns.srct
        dec = suns.write_sun(sidx)
        anorm = accuracy * scene.skyres * (1 - np.cos(.533*np.pi/360))
        super().__init__(scene, stype=f"sun_{sidx:04d}", fdres=fdres,
                         accuracy=anorm, srcdef=dec, **kwargs)
        self.specidx = speclevel - self.idres
        self._keepamb = keepamb
        shape = np.concatenate((self.scene.area.ptshape, self.levels[0]))
        weights = self.pdf_from_sky(SCBinField(self.scene))
        self.weights = translate.resample(weights, shape)

    def __del__(self):
        try:
            if not self._keepamb:
                os.remove(self.compiledscene.replace('.oct', '.amb'))
        except (IOError, TypeError):
            pass
        super().__del__()

    def pdf_from_sky(self, skyfield, interp=12, rebuild=False, zero=True,
                     filterpts=True):
        ishape = np.concatenate((self.scene.area.ptshape,
                                 self.levels[self.specidx-2]))
        fi = f"{self.scene.outdir}/sunpdfidxs.npz"
        if os.path.isfile(fi) and not rebuild:
            f = np.load(fi)
            idxs = f['arr_0']
            errs = f['arr_1']
        else:
            shp = self.levels[self.specidx-2]
            si = np.stack(np.unravel_index(np.arange(np.product(shp)), shp))
            uv = (si.T + .5)/shp[1]
            grid = skyfield.scene.view.uv2xyz(uv)
            idxs, errs = skyfield.query_all_pts(grid, interp)
            strides = np.array(skyfield.lum.index_strides()[:-1])[:, None, None]
            idxs = np.reshape(idxs + strides, (-1, interp))
            errs = errs.reshape(-1, interp)
            np.savez(fi, idxs, errs)
        column = skyfield.lum.full_array()[:, self.sbin]
        if interp > 1:
            lum = np.average(column[idxs], -1, weights=1/errs).reshape(ishape)
        else:
            lum = column[idxs].reshape(ishape)
        if filterpts:
            haspeak = np.max(lum, (2, 3)) > self.srct
            lum = lum * haspeak[..., None, None]
        if zero:
            lum = np.where(lum > self.scene.maxspec, 0, lum)
        return lum

    def sample(self, vecs,
               rcopts='-ab 6 -ad 3000 -as 1500 -st 0 -ss 16 -aa .1',
               nproc=12, ambcache=False, executable='rtrace', **kwargs):
        """call rendering engine to sample sky contribution

        Parameters
        ----------
        vecs: np.array
            shape (N, 6) vectors to calculate contributions for
        rcopts: str, optional
            option string to send to executable
        nproc: int, optional
            number of processes executable should use
        ambcache: bool, optional
            use ambient caching (rcopts should be combatible, appends
            appropriate -af argument)
        executable: str, optional
            path to rendering binary

        Returns
        -------
        lum: np.array
            array of shape (N,) to update weights
        """
        if ambcache:
            afile = self.compiledscene.replace('.oct', '.amb')
            rcopts += f' -af {afile}'
        rc = f"{executable} -fff {rcopts} -h -n {nproc} {self.compiledscene}"
        return super().sample(vecs, call=rc).ravel()

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
        if self.idx == self.specidx:
            skyfield = SCBinField(self.scene)
            shape = np.concatenate((self.scene.area.ptshape,
                                    self.levels[self.idx]))
            weights = self.pdf_from_sky(skyfield)
            p = translate.resample(weights, shape)
            if self.plotp:
                quickplot.imshow(p[0, 0], [20, 10])
            pdraws = draw.from_pdf(p.ravel(), self.slimit)
        else:
            pdraws = super().draw()
        return pdraws
