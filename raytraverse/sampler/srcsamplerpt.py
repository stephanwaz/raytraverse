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
        initialized renderer instance (with scene loaded, including sources)
    source: str
        single scene file containing sources (including sky, lights, sun, etc)
    sunbin: int
        sun bin
    """

    def __init__(self, scene, engine, source, stype="source", scenedetail=False,
                 distance=0.5, normal=5.0, **kwargs):
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

    def run(self, point, posidx, specguide=None, mapper=None, upaxis=2, **kwargs):
        if mapper is None:
            mapper = ViewMapper()
        mapper.jitterrate = 0.8
        self._mapper = mapper
        point = np.asarray(point).flatten()[0:3]
        self._set_normalization(point, 2)
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

    def _set_normalization(self, point, upaxis=2):
        fd, srcoct = tempfile.mkstemp(dir=f"./{self.scene.outdir}/",
                                      prefix='tmp_src')
        with os.fdopen(fd) as f:
            pipeline([f"oconv -w {self.sourcefile}"], outfile=f)
        afac = 0.0
        srcoct = f"./{self.scene.outdir}/" + srcoct.rsplit("/")[-1]
        if self.features > 1:
            vlambda = (1/3, 1/3, 1/3)
        else:
            vlambda = (0.265, 0.670, 0.065)
        if self.lights.size > 0:
            # make rays point at center of sources from 6 cardinal directions
            box = np.vstack((np.eye(3), -np.eye(3))) * .001
            brays = self.lights[:, None, 0:3] + box[None]
            bdirs = np.broadcast_to(-box[None], brays.shape)
            brays = np.concatenate((brays, bdirs), axis=2).reshape(-1, 6)
            # render direct luminance
            icheck = SpRenderer("-h -ab 0 -lw 1e-5 -av 0 0 0 -w "
                                "-dc 1 -ds 0 -dt 0 -dj 0", srcoct, 1)
            lums = np.array([float(i) for i in
                             icheck(brays).split()]).reshape(-1, 3)
            # get the max luminance for each source
            llum = np.max(io.rgb2rad(lums, vlambda).reshape(len(self.lights), -1), 1)
            # find the distance along upaxis
            dray = np.abs(self.lights[:, 0:3] - point[None])[:, upaxis]
            radius = np.sqrt(self.lights[:, 4] / np.pi)
            # add up direct normal irradiance (lum * omega), but cap at max(lum)
            afac += np.min((np.sum(llum * (1 - np.cos(np.arctan(radius/dray)))),
                            np.max(llum)))
        if self.sources.size > 0:
            # get illuminnance of distant sources to also calibrate accuracy
            icheck = SpRenderer("-h -ab 1 -ad 10000 -lw 1e-5 -av 0 0 0 -I+ -w "
                                "-dc 1 -ds 0 -dt 0 -dj 0", srcoct, 1)
            # hopefully this is above any light source geometry
            ray = np.array([0, 0, 1e7, 0, 0, 1]).reshape(-1, 6)
            illum = io.rgb2rad([float(i) for i in icheck(ray).split()], vlambda)
            afac += illum / np.pi
        if afac > 0:
            self.accuracy = self.accuracy * afac
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
                # load reflection normals (from scene.reflection_search)
                normals = io.load_txt(specguide).reshape(-1, 3)
                self._refl = translate.norm(normals)
            except (ValueError, FileNotFoundError):
                pass
        ires = self.idres*180/self._mapper.viewangle
        # 1/2 for nyquist
        level_res = ires*np.power(2, np.arange(self.nlev))/2
        # last level with refinement
        rlevel = len(self._samplelevels) - 2
        if self.sources.size > 0:
            minres = 180/self.sources[:, 3]
            level_idx = np.searchsorted(level_res, minres)
            # allocate direct source sampling to level below nyquist limit
            for idx, src in zip(level_idx, self.sources):
                if self._refl is not None:
                    rsrc = self._reflect(src[None])
                else:
                    rsrc = []
                # ensure atleast 1 levels of source "clean-up" else use
                # a srcviewsampler
                if idx >= rlevel:
                    self._viewdirections.append(src[0:4])
                    self._viewdirections += rsrc
                    self._isdistant += [True]*(len(rsrc) + 1)
                else:
                    self._samplelevels[idx].append(src[0:4])
                    self._samplelevels[idx] += rsrc
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
        self._viewdirections = np.array(self._viewdirections)

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
            srcview = viewsampler.run(point, posidx, vm=vms,
                                      isdistant=self._isdistant)
        lightpoint = LightPointKD(self.scene, self.vecs, self.lum,
                                  src=self.stype, pt=point, write=write,
                                  srcn=self.srcn, posidx=posidx,
                                  vm=vm, srcviews=srcview,
                                  features=outfeatures, **kwargs)
        # reset run() modified parameters
        self.accuracy = self._normaccuracy
        self._viewdirections = []
        self._isdistant = []
        self._samplelevels = [[] for _ in range(self.nlev)]
        self._refl = None
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


