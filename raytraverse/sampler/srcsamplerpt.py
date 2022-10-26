# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import numpy as np

from raytraverse import translate, io
from raytraverse.lightpoint import LightPointKD
from raytraverse.mapper import ViewMapper
from raytraverse.sampler.samplerpt import SamplerPt
from raytraverse.sampler.srcsamplerptview import SrcSamplerPtView
from raytraverse.sampler import draw
from raytraverse.sampler.basesampler import filterdict


class SrcSamplerPt(SamplerPt):
    """sample contributions from fixed sources.

    Parameters
    ----------
    scene: raytraverse.scene.Scene
        scene class containing geometry, location and analysis plane
    engine: raytraverse.renderer.Rtrace
        initialized renderer instance (with scene loaded, including sources)
    source: str
        single scene file containing sources (including sky, lights, sun, etc)
    sunbin: int
        sun bin
    """

    def __init__(self, scene, engine, source, stype="source", scenedetail=False,
                 distance=0.5, normal=5.0, t0=20, t1=400, **kwargs):
        self._scenedetail = scenedetail
        self._distance = distance
        self._normal = normal * np.pi/180
        self._oospec = engine.ospec
        if scenedetail:
            # use deterministic sampler (take all points above threshold
            # otherwise luminance detail gets undersanmpled
            self.ub = 1
            engine.update_ospec(engine.ospec + "LNM")
        kwargs.update(features=engine.features)
        super().__init__(scene, engine, stype=stype, t0=t0/179,
                         t1=t1/179, **kwargs)
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
        #: set sampling level/strategy for lights based on source solid angle,
        #: fill ratio and sampling resolution
        self._samplelevels = [[] for _ in range(self.nlev)]

    def run(self, point, posidx, specguide=None, mapper=None, upaxis=2,
            **kwargs):
        if mapper is None:
            mapper = ViewMapper()
        self._mapper = mapper
        point = np.asarray(point).flatten()[0:3]
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
            p = np.ones(len(pdraws))
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
                    if self.features > 1:
                        insrc = np.tile(insrc, self.weights.shape[0])
                    # definite draw on source region
                    p[0:len(insrc)][insrc] = self.t1 * 2 * self.ub
            if self.features > 1:
                p = self.featurefunc(p.reshape(self.weights.shape),
                                     axis=0).ravel()
            pdraws = draw.from_pdf(p, self._threshold(level),
                                   lb=self.lb, ub=self.ub, minsamp=1)
        return pdraws, p

    def _process_features(self, lum):
        if self._scenedetail:
            slum = lum[:, :-5]
            # scale to definite draw at thresholds
            afac = self.t1 * self.accuracy
            d = lum[:, -5:-4] * afac / self._distance
            n = np.arccos(lum[:, -4:-1]) * afac / self._normal
            # always draw on material difference
            m = lum[:, -1:] * 2 * afac
            dlum = np.hstack((slum, d, n, m))
            return slum, dlum
        else:
            return super()._process_features(lum)

    def _load_specguide(self, point, specguide):
        """
        Parameters
        ----------
        specguide: str
            file with reflection normals.
        """
        normal = None
        if hasattr(specguide, "lower"):
            try:
                # load reflection normals (from scene.reflection_search)
                normal = translate.norm(io.load_txt(specguide).reshape(-1, 3))
            except (ValueError, FileNotFoundError):
                pass
        ires = self.idres*180/self._mapper.viewangle
        # 1/2 for nyquist
        level_res = ires*np.power(2, np.arange(self.nlev))/2
        # last level with refinement
        rlevel = max(0, len(self._samplelevels) - 2)
        # allocate direct source sampling to level below nyquist limit
        vd = []
        # first add sources
        for src in self.sources:
            res = 180/src[3]
            idx = np.searchsorted(level_res, res)
            # always viewsample sunlike sources
            if res > level_res[rlevel] or src[3] < 2:
                vd.append(src[0:4])
            elif res > level_res[0]:
                self._samplelevels[idx].append(src[0:4])
        vd = np.array(vd)
        # then add unique reflections
        for src in self.sources:
            res = 180/src[3]
            if normal is not None and res > level_res[rlevel]:
                rsrc = list(translate.reflect(src[None, 0:3], normal, True))
                for r in rsrc:
                    if not np.any(np.all(np.isclose(r, vd[:, 0:3]), 1)):
                        vd = np.concatenate((vd, [(*r, src[3] * 2)]))
        if self.lights.size > 0:
            # distance to source
            lightdist = np.linalg.norm(self.lights[:, 0:3] - point[None],
                                       axis=1)
            # direction to source
            lightdir = translate.norm(self.lights[:, 0:3] - point[None])
            # apparent size (degrees diameter) of light
            appsize = np.arctan(self.lights[:, 3]/lightdist)*360/np.pi
            # aspect ratio of light
            aterm = 4 - np.minimum(4, np.pi**2*np.square(self._fillratio))
            # minimum side
            minside = np.sqrt(2 - np.sqrt(aterm))*self.lights[:, 3]/lightdist
            # apparent size (degrees diameter) of min-side
            minsize = np.arctan(minside*0.5)*360/np.pi
            minres = 180/minsize
            level_idx = np.searchsorted(level_res, minres)
            # allocate direct source sampling to level below nyquist limit
            for idx, light, size in zip(level_idx, lightdir, appsize):
                li = np.concatenate((light, [size]))
                self._samplelevels[min(idx, rlevel)].append(li)
        self._viewdirections = vd

    def _run_callback(self, point, posidx, vm, write=True, **kwargs):
        if self._scenedetail:
            outfeatures = self.engine.update_ospec(self._oospec)
        else:
            outfeatures = self.engine.features
        if self._viewdirections is None or len(self._viewdirections) == 0:
            srcview = None
        else:
            viewsampler = SrcSamplerPtView(self.scene, self.engine,
                                           samplerlevel=self._slevel + 1)
            vms = [ViewMapper(j[0:3], j[3], name=f"{self.stype}_{i}",
                              jitterrate=0)
                   for i, j in enumerate(self._viewdirections)]
            srcview = viewsampler.run(point, posidx, vm=vms)
        lightpoint = LightPointKD(self.scene, self.vecs, self.lum,
                                  src=self.stype, pt=point, write=write,
                                  srcn=self.srcn, posidx=posidx,
                                  vm=vm, srcviews=srcview,
                                  features=outfeatures, **kwargs)
        # reset run() modified parameters
        self._viewdirections = []
        self._samplelevels = [[] for _ in range(self.nlev)]
        return lightpoint

    def _plot_weights(self, level, vm, name, suffix=".hdr", fisheye=True):
        outw = (f"{self.scene.outdir}_{name}_{self.stype}_weights_"
                f"{level:02d}{suffix}")
        if self.features > 1:
            if self.features == 3:
                fweights = ['red', 'green', 'blue']
            elif self.features == 6:
                fweights = ['lum', 'distance', 'nx', 'ny', 'nz', 'material']
            else:
                fweights = ['red', 'green', 'blue', 'distance', 'nx', 'ny',
                            'nz', 'material']
            for i, fw in enumerate(fweights):
                of = outw.replace("_weights_", f'_weight_{fw}_')
                self._plot_dist(self.weights[i], vm, of, fisheye)
        else:
            w = self.weights
            self._plot_dist(w, vm, outw, fisheye)


