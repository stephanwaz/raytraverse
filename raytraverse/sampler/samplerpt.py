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
        initial direction resolution (as log2(res))
    fdres: int, optional
        final directional resolution given as log2(res)
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
    bands: int, optional
        number of spectral bands returned by the engine
    engine_args: str, optional
        command line arguments used to initialize engine
    nproc: int, optional
        number of processors to give to the engine, if None, uses os.cpu_count()
    """

    def __init__(self, scene, engine, idres=5, fdres=9,
                 accuracy=1.0,  srcn=1, stype='generic', bands=1, **kwargs):
        #: int: number of spectral bands / channels returned by renderer
        #: based on given renderopts (user ensures these agree).
        self.bands = bands
        #: int: number of sources return per vector by run
        self.srcn = srcn
        #: int: initial direction resolution (as log2(res))
        self.idres = idres
        self.fdres = fdres
        super().__init__(scene, engine, accuracy=accuracy, stype=stype)

    def sampling_scheme(self, a):
        """calculate sampling scheme"""
        return np.array([(2**i*a, 2**i)
                         for i in range(self.idres, self.fdres+1, 1)])

    def run(self, point, posidx, mapper=None, **kwargs):
        """sample a single point, poisition index handles file naming

        Parameters
        ----------
        point: np.array
            point to sample
        posidx: int
            position index
        mapper: raytraverse.mapper.ViewMapper
            view direction to sample
        kwargs:
            passed to BaseSampler.run()

        Returns
        -------
        LightPointKD
        """
        if mapper is None:
            mapper = ViewMapper()
        point = np.asarray(point).flatten()[0:3]
        mapper.origin = point
        name = f"{mapper.name}_{posidx:06d}"
        levels = self.sampling_scheme(mapper.aspect)
        super().run(mapper, name, levels, **kwargs)
        return self._run_callback(point, posidx, mapper)

    def _run_callback(self, point, posidx, vm, write=True):
        """handle class specific lightpointKD construction"""
        lightpoint = LightPointKD(self.scene, self.vecs, self.lum,
                                  src=self.stype, pt=point, write=write,
                                  srcn=self.srcn, posidx=posidx, vm=vm)
        return lightpoint

    def _plot_p(self, p, level, vm, name, suffix=".hdr", fisheye=True):
        ps = p.reshape(self.weights.shape)
        outshape = (512*vm.aspect, 512)
        res = outshape[-1]
        outw = (f"{self.scene.outdir}_{name}_{self.stype}_weights_"
                f"{level:02d}{suffix}")
        outp = (f"{self.scene.outdir}_{name}_{self.stype}_detail_"
                f"{level:02d}{suffix}")
        if fisheye:
            pixelxyz = vm.pixelrays(res)
            uv = vm.xyz2uv(pixelxyz.reshape(-1, 3))
            pdirs = np.concatenate((pixelxyz[0:res], -pixelxyz[res:]), 0)
            mask = vm.in_view(pdirs, indices=False).reshape(outshape)
            ij = translate.uv2ij(uv, self.weights.shape[-1], aspect=vm.aspect)
            img = self.weights[ij[:, 0], ij[:, 1]].reshape(outshape)
            io.array2hdr(np.where(mask, img, 0), outw)
            img = ps[ij[:, 0], ij[:, 1]].reshape(outshape)
            io.array2hdr(np.where(mask, img, 0), outp)
        else:
            weights = translate.resample(self.weights[-1::-1], outshape,
                                         radius=0, gauss=False)
            detail = translate.resample(ps[-1::-1], outshape, radius=0,
                                        gauss=False)
            if vm.aspect == 2:
                weights = np.concatenate((weights[res:], weights[0:res]), 0)
                detail = np.concatenate((detail[res:], detail[0:res]), 0)
            io.array2hdr(weights, outw)
            io.array2hdr(detail, outp)

    def _plot_vecs(self, vecs, level, vm, name, suffix=".hdr", fisheye=True):
        outshape = (512*vm.aspect, 512)
        outf = (f"{self.scene.outdir}_{name}_{self.stype}_samples_"
                f"{level:02d}{suffix}")
        img = np.zeros(outshape)
        img = vm.add_vecs_to_img(img, vecs[:, 3:], channels=level+1, grow=1,
                                 fisheye=fisheye)
        io.array2hdr(img, outf)