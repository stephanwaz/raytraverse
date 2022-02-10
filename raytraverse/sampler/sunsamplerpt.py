# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import os
import tempfile

import numpy as np

from raytraverse import translate, io
from raytraverse.lightpoint import LightPointKD
from raytraverse.mapper import ViewMapper
from raytraverse.sampler.samplerpt import SamplerPt
from raytraverse.sampler.sunsamplerptview import SunSamplerPtView
from raytraverse.sampler import draw


class SunSamplerPt(SamplerPt):
    """sample contributions from direct suns.

    Parameters
    ----------
    scene: raytraverse.scene.Scene
        scene class containing geometry, location and analysis plane
    engine: raytraverse.renderer.Rtrace
        initialized renderer instance (with scene loaded, no sources)
    sun: np.array
        shape 3, sun position
    sunbin: int
        sun bin
    speclevel: int, optional
        at this sampling level, pdf is made from brightness of sky sampling
        rather than progressive variance to look for fine scale specular
        highlights, this should be atleast 1 level from the end and the
        resolution of this level should be smaller than the size of the source
    slimit: float, optional
        the minimum value in the specular guide considered as a potential
        specular reflection source, in the case of low vlt glazing, this
        value should be reduced.
    maxspec: float, optional
        the maximum value inn the specular guide considered as a specular
        reflection source. above this value it is assumed that these are direct
        view rays to the source so are not sampled. in the case of low vlt
        glazing, this value should be reduced. In mixed (high-low) vlt scenes
        the specular guide will either over sample (including direct views) or
        under sample (miss specular reflections) depending on this setting.
    """

    def __init__(self, scene, engine, sun, sunbin, speclevel=9, fdres=10,
                 slimit=0.01, maxspec=0.2, stype='sun', **kwargs):
        self.slimit = slimit
        self.maxspec = maxspec
        self._specguide = None
        super().__init__(scene, engine, stype=f"{stype}_{sunbin:04d}",
                         fdres=fdres, **kwargs)
        # update parameters post init
        # normalize accuracy for sun source
        self.accuracy = self.accuracy * (1 - np.cos(.533*np.pi/360))
        #: int: index of level at which brightness sampling occurs
        self.specidx = min(speclevel, fdres) - self.idres
        #: np.array: sun position x,y,z
        self.sunpos = np.asarray(sun).flatten()[0:3]
        self.sunbin = sunbin
        # load new source
        f, srcdef = tempfile.mkstemp(dir=f"./{scene.outdir}/", prefix='tmp_src')
        # srcdef = f'{scene.outdir}/tmp_srcdef_{sunbin}.rad'
        f = open(srcdef, 'w')
        f.write(scene.formatter.get_sundef(sun, (1, 1, 1)))
        f.close()
        ambfile = f"{scene.outdir}/{stype}_{sunbin:04d}.amb"
        self.engine.load_source(srcdef, ambfile=ambfile)
        os.remove(srcdef)

    def run(self, point, posidx, vm=None, plotp=False, specguide=None,
            **kwargs):
        if vm is None:
            vm = ViewMapper()
        self._levels = self.sampling_scheme(vm.aspect)
        self._load_specguide(specguide, vm)
        if plotp:
            io.array2hdr(self._specguide, "specguide.hdr")
        return super().run(point, posidx, vm, plotp=plotp, **kwargs)

    def draw(self, level):
        """draw samples based on detail calculated from weights

        Returns
        -------
        pdraws: np.array
            index array of flattened samples chosen to sample at next level
        """
        pdraws, pa = super().draw(level)
        s = 0
        if level == self.specidx and self._specguide is not None:
            s = self._specguide.ravel()
            s[pdraws] = 0
            # tried lowering slimit to capture small sun patches, but got
            # large positive bias...
            # sp = np.percentile(s, 99)
            # if self.slimit < sp:
            #     slimit = self.slimit
            # else:
            #     slimit = max(sp, self.slimit/100)
            sdraws = draw.from_pdf(s, self.slimit, ub=1)
            pdraws = np.concatenate((pdraws, sdraws))
        return pdraws, pa + s

    def _load_specguide(self, specguide, vm):
        """

        Parameters
        ----------
        specguide: raytraverse.lightpoint.LightPointKD
        vm: raytraverse.mapper.ViewMappper
        """
        if specguide is None:
            self._specguide = None
        else:
            if not hasattr(specguide, '__len__'):
                specguide = [specguide]
            details = []
            for skykd in specguide:
                side = int(np.sqrt(skykd.srcn - 1))
                skybin = translate.xyz2skybin(self.sunpos, side, tol=.125)
                shp = self.levels[self.specidx]
                si = np.stack(np.unravel_index(np.arange(np.product(shp)), shp))
                uv = (si.T + .5)/shp[1]
                grid = vm.uv2xyz(uv)
                i = skykd.query_ray(grid)[0]
                lumg = np.max(skykd.lum[:, skybin], 1)[i].reshape(shp)
                details.append(np.where(lumg > self.maxspec, 0, lumg))
            self._specguide = np.max(details, 0)

    def _run_callback(self, point, posidx, vm, write=True, **kwargs):
        args = self.engine.args
        # temporarily override arguments
        self.engine.set_args(self.engine.directargs)
        viewsampler = SunSamplerPtView(self.scene, self.engine, self.sunpos,
                                       self.sunbin,
                                       samplerlevel=self._slevel + 1)
        sunview = viewsampler.run(point, posidx)
        self.engine.set_args(args)
        lightpoint = LightPointKD(self.scene, self.vecs, self.lum,
                                  srcdir=self.sunpos, src=self.stype, pt=point,
                                  write=write, srcn=self.srcn, posidx=posidx,
                                  vm=vm, srcviews=[sunview], **kwargs)
        return lightpoint


