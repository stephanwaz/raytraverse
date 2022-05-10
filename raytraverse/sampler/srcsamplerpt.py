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
from clasp.script_tools import pipeline

from raytraverse import translate, io
from raytraverse.lightpoint import LightPointKD
from raytraverse.mapper import ViewMapper
from raytraverse.sampler.samplerpt import SamplerPt
from raytraverse.sampler.srcsamplerptview import SrcSamplerPtView
from raytraverse.renderer import SpRenderer
from raytraverse.sampler import draw
from raytraverse.sampler.basesampler import filterdict


class SrcSamplerPt(SamplerPt):
    """sample contributions from fixed sources.

    Parameters
    ----------
    scene: raytraverse.scene.Scene
        scene class containing geometry, location and analysis plane
    engine: raytraverse.renderer.Rtrace
        initialized renderer instance (with scene loaded, no sources)
    source: str
        single scene file containing sources (including sky, lights, sun, etc)
    sunbin: int
        sun bin
    """

    def __init__(self, scene, engine, source, stype="source", **kwargs):
        super().__init__(scene, engine, stype=stype, **kwargs)
        # update parameters post init
        #: path to source scene file
        self.sourcefile = source
        srcs, distant = self.engine.get_sources()
        #: non distant sources, pos, radius, area
        self.lights = srcs[np.logical_not(distant)]
        #: amount of circle filled by each light
        self._fillratio = (self.lights[:, 4] /
                           (np.square(self.lights[:, 3]) * np.pi))
        #: distant sources (includes those with large solid angles
        self.sources = srcs[distant]
        #: gets initialized for each point, as apparent light size will change
        self._viewdirections = []
        self._isdistant = []
        #: gets initialized for each point using direct illuminance from sources
        #: should not be less than 1.08173E-05 (accuracy of sunsampler pt)
        self._normaccuracy = self.accuracy
        #: set sampling level/strategy for lights based on source solid angle,
        #: fill ratio and sampling resolution
        self._samplelevels = [[] for i in range(self.nlev)]
        self._refl = None

    def run(self, point, posidx, specguide=None, mapper=None, **kwargs):
        if mapper is None:
            mapper = ViewMapper()
        mapper.jitterrate = 0.8
        self._mapper = mapper
        point = np.asarray(point).flatten()[0:3]
        self._set_normalization(point)
        self._load_specguide(point, specguide)
        return super().run(point, posidx, mapper=mapper, **kwargs)

    def draw(self, level):
        """draw samples based on detail calculated from weights

        Returns
        -------
        pdraws: np.array
            index array of flattened samples chosen to sample at next level
        p: np.array
            computed probabilities
        """
        dres = self.levels[level]
        # sample all if weights is not set or all even
        if level == 0 and np.var(self.weights) < 1e-9:
            pdraws = np.arange(int(np.prod(dres)))
            p = np.ones(self.weights.shape)
        else:
            # use weights directly on first pass
            if level == 0:
                p = self.weights.ravel()
            else:
                p = draw.get_detail(self.weights, *filterdict[self.detailfunc])
                for src in self._samplelevels[level]:
                    vm = ViewMapper(src[0:3], src[3])
                    uv = self._mapper.idx2uv(np.arange(int(np.prod(dres))),
                                             dres, False)
                    insrc = vm.in_view(self._mapper.uv2xyz(uv), False)
                    p[insrc] = self.t1 * 2 * self.ub
            # draw on pdf
            pdraws = draw.from_pdf(p, self._threshold(level),
                                   lb=self.lb, ub=self.ub)
        return pdraws, p

    def _set_normalization(self, point):
        f, srcoct = tempfile.mkstemp(dir=f"./{self.scene.outdir}/",
                                     prefix='tmp_src')
        pipeline([f"oconv -w {self.sourcefile}"], outfile=srcoct,
                 writemode='wb')
        icheck = SpRenderer("-h -ab 1 -ad 10000 -lw 1e-5 -av 0 0 0 -I+ -w "
                            "-dc 1 -ds 0 -dt 0 -dj 0", srcoct, 1)
        ray = np.concatenate((point, [0, 0, 1])).reshape(-1, 6)
        illum = io.rgb2rad([float(i) for i in icheck(ray).split()])
        self.accuracy = self.accuracy * max(1.08173E-05, illum/np.pi)
        os.remove(srcoct)

    def _reflect(self, sources):
        srcr, m = translate.reflect(sources[:, 0:3],
                                    self._refl, False)
        if len(sources) > 1:
            areas = np.broadcast_to(sources[:, 3:4],
                                    m.shape)[m, None]
        else:
            areas = np.full(len(m), sources[0, 3])[m, None]
        srcr = srcr[m]
        return list(np.hstack((srcr, areas)))

    def _load_specguide(self, point, specguide):
        """
        Parameters
        ----------
        specguide: str
            file with reflection normals.
        """
        if hasattr(specguide, "lower"):
            try:
                self._refl = translate.norm(io.load_txt(specguide).reshape(-1, 3))
            except (ValueError, FileNotFoundError):
                pass
        ires = self.idres*180/self._mapper.viewangle
        # 1/2 for nyquist
        level_res = ires*np.power(2, np.arange(self.nlev))/2
        if self.sources.size > 0:
            minres = 180/self.sources[:, 3]
            level_idx = np.searchsorted(level_res, minres)
            for idx, src in zip(level_idx, self.sources):
                if self._refl is not None:
                    rsrc = self._reflect(src[None])
                else:
                    rsrc = []
                if idx >= len(self._samplelevels) - 2:
                    self._viewdirections.append(src[0:4])
                    self._viewdirections += rsrc
                    self._isdistant += [True]*(len(rsrc) + 1)
                else:
                    self._samplelevels[idx].append(src[0:4])
                    self._samplelevels[idx] += rsrc
        if self.lights.size > 0:
            lightdist = np.linalg.norm(self.lights[:, 0:3] - point[None],
                                       axis=1)
            lightdir = translate.norm(self.lights[:, 0:3] - point[None])
            appsize = np.arctan(self.lights[:, 3]/lightdist)*360/np.pi
            aterm = 4 - np.minimum(4, np.pi**2*np.square(self._fillratio))
            minside = np.sqrt(2 - np.sqrt(aterm))*self.lights[:, 3]/lightdist
            minsize = np.arctan(minside*0.5)*360/np.pi
            minres = 180/minsize
            level_idx = np.searchsorted(level_res, minres)
            for idx, light, size in zip(level_idx, lightdir, appsize):
                li = np.concatenate((light, [size]))
                if idx >= len(self._samplelevels):
                    self._viewdirections.append(li)
                    self._isdistant.append(False)
                else:
                    self._samplelevels[idx].append(li)
        self._viewdirections = np.array(self._viewdirections)

    def _run_callback(self, point, posidx, vm, write=True, **kwargs):
        if self._viewdirections is None or len(self._viewdirections) == 0:
            srcview = None
        else:
            viewsampler = SrcSamplerPtView(self.scene, self.engine,
                                           samplerlevel=self._slevel + 1)
            vms = [ViewMapper(j[0:3], j[3], name=f"{self.stype}_{i}",
                              jitterrate=0)
                   for i, j in enumerate(self._viewdirections)]
            srcview = viewsampler.run(point, posidx, vm=vms,
                                      isdistant=self._isdistant)
        lightpoint = LightPointKD(self.scene, self.vecs, self.lum,
                                  src=self.stype, pt=point, write=write,
                                  srcn=self.srcn, posidx=posidx,
                                  vm=vm, srcviews=srcview, **kwargs)
        self.accuracy = self._normaccuracy
        self._viewdirections = []
        self._isdistant = []
        self._samplelevels = [[] for i in range(self.nlev)]
        self._refl = None
        return lightpoint


