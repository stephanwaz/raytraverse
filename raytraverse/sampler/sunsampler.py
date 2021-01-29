# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import os

import numpy as np

from raytraverse import translate, draw, io
from raytraverse.lightpoint import LightPointKD, SunPointKD
from raytraverse.mapper import ViewMapper
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
                 engine_args='-ab 7 -ad 10 -c 100 -as 0 -lw 1.25e-5',
                 keepamb=True, ambcache=False, slimit=0.01, maxspec=0.3,
                 **kwargs):
        self.slimit = slimit
        self.maxspec = maxspec
        self.specguide = None
        # update ambient file and args before init
        self._keepamb = keepamb and ambcache
        if ambcache:
            self.ambfile = f"{scene.outdir}/sun_{sunbin:04d}.amb"
        else:
            self.ambfile = None
        engine_args = scene.formatter.get_standard_args(engine_args,
                                                        self.ambfile)
        super().__init__(scene, stype=f"sun_{sunbin:04d}", fdres=fdres,
                         engine_args=engine_args, **kwargs)
        # update parameters post init
        # normalize accuracy for sun source
        self.accuracy = self.accuracy * (1 - np.cos(.533*np.pi/360))
        #: int: index of level at which brightness sampling occurs
        self.specidx = speclevel - self.idres
        #: np.array: sun position x,y,z
        self.sunpos = np.asarray(sun).flatten()[0:3]

        # load new source
        srcdef = f'{scene.outdir}/tmp_srcdef_{sunbin}.rad'
        f = open(srcdef, 'w')
        f.write(scene.formatter.get_sundef(sun, (1, 1, 1)))
        f.close()
        self.engine.load_source(srcdef)
        os.remove(srcdef)

    def sample(self, vecf, vecs, outf=None):
        """call rendering engine to sample sky contribution"""
        return super().sample(vecf, vecs, outf=outf).ravel()

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

    def run_callback(self, vecfs, name, point, posidx, vm):
        outf = f'{self.scene.outdir}/{name}_{self.stype}_vals.out'
        vecs = []
        for vecf in vecfs:
            fsrc = open(vecf, 'rb')
            vecs.append(io.bytefile2np(fsrc, (-1, 6)))
            fsrc.close()
        vecs = np.concatenate(vecs)
        lightpoint = SunPointKD(self.scene, vecs, outf, sun=self.sunpos,
                                src=self.stype, pt=point, write=True,
                                srcn=self.srcn, posidx=posidx, vm=vm)
        [os.remove(vecf) for vecf in vecfs]
        if not self._keepamb:
            try:
                os.remove(self.ambfile)
            except (IOError, TypeError):
                pass
        return lightpoint

    def _load_specguide(self, point, posidx, vm):
        try:
            skykd = LightPointKD(self.scene, pt=point, posidx=posidx, src='sky')
        except ValueError:
            self.specguide = None
        else:
            side = int(np.sqrt(skykd.srcn - 1))
            skybin = translate.xyz2skybin(self.sunpos, side, tol=.125)
            shp = self.levels[self.specidx]
            si = np.stack(np.unravel_index(np.arange(np.product(shp)), shp))
            uv = (si.T + .5)/shp[1]
            grid = vm.uv2xyz(uv)
            i = skykd.query_ray(grid)[0]
            lumg = np.max(skykd.lum[:, skybin], 1)[i].reshape(shp)
            self.specguide = np.where(lumg > self.maxspec, 0, lumg)

    def run(self, point, posidx, vm=None, plotp=False, **kwargs):
        if vm is None:
            vm = ViewMapper()
        self._load_specguide(point, posidx, vm)
        if plotp:
            io.array2hdr(self.specguide, "specguide.hdr")
        return super().run(point, posidx, vm, plotp, **kwargs)
