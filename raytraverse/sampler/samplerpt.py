# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import numpy as np

from raytraverse import translate, io
from raytraverse.sampler.basesampler import BaseSampler
from raytraverse.lightpoint import LightPointKD
from raytraverse.mapper import ViewMapper


class SamplerPt(BaseSampler):
    """wavelet based sampling class for direction rays from a point

    Parameters
    ----------
    scene: raytraverse.scene.Scene
        scene class containing geometry and formatter compatible with engine
    engine: raytraverse.renderer.Renderer
        should inherit from raytraverse.renderer.Renderer
    idres: int, optional
        initial direction resolution (as sqrt of samples per hemisphere)
    nlev: int, optional
        number of levels to sample (eeach lvl doubles idres)
    accuracy: float, optional
        parameter to set threshold at sampling level relative to final level
        threshold (smaller number will increase sampling, default is 1.0)
    srcn: int, optional
        number of sources return per vector by run
    stype: str, optional
        sampler type (prefixes output files)
    srcdef: str, optional
        path or string with source definition to add to scene
    plotp: bool, optional
        show probability distribution plots at each level (first point only)
    features: int, optional
        number of values evaluated for detail
    engine_args: str, optional
        command line arguments used to initialize engine
    nproc: int, optional
        number of processors to give to the engine, if None, uses os.cpu_count()
    """

    _includeorigin = True

    def __init__(self, scene, engine, idres=32, nlev=5, accuracy=1.0,
                 srcn=1, stype='generic', features=1, samplerlevel=0, **kwargs):
        #: int: number of sources return per vector by run
        self.srcn = srcn
        #: int: initial direction resolution (as sqrt of samples per hemisphere
        #: (or view angle)
        self.idres = idres
        self.nlev = nlev
        super().__init__(scene, engine, accuracy=accuracy, stype=stype,
                         samplerlevel=samplerlevel, features=features, **kwargs)

    def sampling_scheme(self, a):
        """calculate sampling scheme"""
        return np.array([(self.idres*2**i*a, self.idres*2**i)
                         for i in range(self.nlev)])

    def run(self, point, posidx, mapper=None, lpargs=None, **kwargs):
        """sample a single point, position index handles file naming

        Parameters
        ----------
        point: np.array
            point to sample
        posidx: int
            position index
        mapper: raytraverse.mapper.ViewMapper
            view direction to sample
        lpargs: dict, optional
            keyword arguments forwarded to LightPointKD construction
        kwargs:
            passed to BaseSampler.run()

        Returns
        -------
        LightPointKD
        """
        if lpargs is None:
            lpargs = {}
        if mapper is None:
            mapper = ViewMapper()
        point = np.asarray(point).flatten()[0:3]
        mapper.origin = point
        name = f"{mapper.name}_{posidx:06d}"
        levels = self.sampling_scheme(mapper.aspect)
        super().run(mapper, name, levels, **kwargs)
        return self._run_callback(point, posidx, mapper, **lpargs)

    def repeat(self, guide, stype):
        ostype = self.stype
        self.stype = stype

        mapper = guide.vm
        mapper.origin = guide.pt

        self.vecs = None
        self.lum = []

        gvecs = guide.vec
        vecs = np.hstack((np.broadcast_to(mapper.origin, gvecs.shape), gvecs))
        self.sample(vecs)
        lp = self._run_callback(guide.pt, guide.posidx, mapper,
                                parent=guide.parent)
        self.stype = ostype
        return lp

    def _run_callback(self, point, posidx, vm, write=True, **kwargs):
        """handle class specific lightpointKD construction"""
        lightpoint = LightPointKD(self.scene, self.vecs, self.lum,
                                  src=self.stype, pt=point, write=write,
                                  srcn=self.srcn, posidx=posidx, vm=vm,
                                  features=self.engine.features, **kwargs)
        return lightpoint

    @staticmethod
    def _plot_dist(ps, vm, outf, fisheye=True):
        outshape = (512*vm.aspect, 512)
        res = outshape[-1]
        if fisheye:
            pixelxyz = vm.pixelrays(res)
            uv = vm.xyz2uv(pixelxyz.reshape(-1, 3))
            pdirs = np.concatenate((pixelxyz[0:res], -pixelxyz[res:]), 0)
            mask = vm.in_view(pdirs, indices=False).reshape(outshape)
            ij = translate.uv2ij(uv, ps.shape[-1], aspect=vm.aspect)
            img = ps[ij[:, 0], ij[:, 1]].reshape(outshape)
            io.array2hdr(np.where(mask, img, 0), outf)
        else:
            detail = translate.resample(ps[-1::-1], outshape, radius=0,
                                        gauss=False)
            io.array2hdr(detail, outf)

    def _plot_p(self, p, level, vm, name, suffix=".hdr", fisheye=True):
        ps = p.reshape(self.weights.shape[-2:])
        outp = (f"{self.scene.outdir}_{name}_{self.stype}_detail_"
                f"{level:02d}{suffix}")
        self._plot_dist(ps, vm, outp, fisheye)

    def _plot_weights(self, level, vm, name, suffix=".hdr", fisheye=True):
        outw = (f"{self.scene.outdir}_{name}_{self.stype}_weights_"
                f"{level:02d}{suffix}")
        if self.features > 1:
            w = np.average(self.weights, 0)
        else:
            w = self.weights
        self._plot_dist(w, vm, outw, fisheye)

    def _plot_vecs(self, vecs, level, vm, name, suffix=".hdr", fisheye=True):
        outshape = (512*vm.aspect, 512)
        outf = (f"{self.scene.outdir}_{name}_{self.stype}_samples_"
                f"{level:02d}{suffix}")
        img = np.zeros(outshape)
        img = vm.add_vecs_to_img(img, vecs[:, 3:], channels=level+1, grow=1,
                                 fisheye=fisheye)
        io.array2hdr(img, outf)
