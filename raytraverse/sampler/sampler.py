# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import os

import numpy as np

from raytraverse import translate, io, renderer
from raytraverse.sampler import draw
from raytraverse.lightpoint import LightPointKD
from raytraverse.mapper import ViewMapper


class Sampler(object):
    """wavelet based sampling class

    Parameters
    ----------
    scene: raytraverse.scene.Scene
        scene class containing geometry and formatter compatible with engine
    engine: type, optional
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

    #: coefficients used to set the sampling thresholds
    t0 = 2**-8
    t1 = .0625

    #: lower and upper bounds for drawing from pdf
    lb = .25
    ub = 8

    def __init__(self, scene, engine=renderer.Rtrace, idres=5, fdres=9,
                 accuracy=1.0,  srcn=1, stype='generic', srcdef=None, bands=1,
                 engine_args="", nproc=None, **kwargs):
        #: raytraverse.renderer.Renderer
        self.engine = engine()
        self._staticscene = True
        #: int: number of spectral bands / channels returned by renderer
        #: based on given renderopts (user ensures these agree).
        self.bands = bands
        #: raytraverse.scene.Scene: scene information
        self.scene = scene
        #: int: number of sources return per vector by run
        self.srcn = srcn
        #: float: accuracy parameter
        self.accuracy = accuracy
        #: int: initial direction resolution (as log2(res))
        self.idres = idres
        self.fdres = fdres
        self.levels = 2
        #: np.array: holds weights for self.draw
        self.weights = np.empty(0)
        #: str: sampler type
        self.stype = stype
        self.compiledscene = srcdef
        self.engine_args = (engine_args, self.compiledscene)
        self.engine_kwargs = dict(nproc=nproc, iot="ff")

    def __del__(self):
        if not self._staticscene:
            try:
                os.remove(self.compiledscene)
            except (IOError, TypeError):
                pass

    @property
    def compiledscene(self):
        return self._compiledscene

    @compiledscene.setter
    def compiledscene(self, src):
        self._staticscene = src is None
        if self._staticscene:
            self._compiledscene = self.scene.scene
        else:
            self._compiledscene = f'{self.scene.outdir}/{self.stype}.oct'
            self.scene.formatter.add_source(self.scene.scene, src,
                                            self._compiledscene)

    @property
    def levels(self):
        """sampling scheme

        :getter: Returns the sampling scheme
        :setter: Set the sampling scheme from (ptres, fdres, skres)
        :type: np.array
        """
        return self._levels

    @levels.setter
    def levels(self, a):
        """calculate sampling scheme"""
        self._levels = np.array([(2**i*a, 2**i)
                                 for i in range(self.idres, self.fdres+1, 1)])

    def sample(self, vecf, vecs, outf=None):
        """generic sample function

        Parameters
        ----------
        vecf: str
            path of file name with sample vectors
            shape (N, 6) vectors in binary float format
        vecs: np.array
            sample vectors (subclasses can choose which to use)
        outf: str, optional
            if given, append results to file

        Returns
        -------
        lum: np.array
            array of shape (N,) to update weights
        """
        if outf is not None:
            f = open(outf, 'a+b')
            lumb = self.engine.call(vecf, outf=f)
            f.close()
        else:
            lumb = self.engine.call(vecf)
        shape = (-1, self.srcn, self.bands)
        lum = io.bytes2np(lumb, shape)
        return np.squeeze(lum, axis=-1)

    def _offset(self, shape, dim):
        """for modifying jitter behavior of UV direction samples

        Parameters
        ----------
        shape: tuple
            shape of samples to jitter/offset
        dim: int
            number of divisions in square side
        """
        return np.random.default_rng().random(shape)/dim

    def sample_to_uv(self, pdraws, shape):
        """generate samples vectors from flat draw indices

        Parameters
        ----------
        pdraws: np.array
            flat index positions of samples to generate
        shape: tuple
            shape of level samples

        Returns
        -------
        si: np.array
            index array of draws matching samps.shape
        vecs: np.array
            sample vectors
        """
        if len(pdraws) == 0:
            return [], []
        # index assignment
        si = np.stack(np.unravel_index(pdraws, shape))
        # convert to UV directions and positions
        uv = si.T/shape[1]
        uv += self._offset(uv.shape, shape[1])
        return si, uv

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
            io.array2hdr(translate.resample(self.weights[-1::-1], outshape,
                                            radius=0), outw)
            io.array2hdr(translate.resample(ps[-1::-1], outshape,
                                            radius=0), outp)

    def _plot_vecs(self, vecs, level, vm, name, suffix=".hdr"):
        outshape = (512*vm.aspect, 512)
        outf = (f"{self.scene.outdir}_{name}_{self.stype}_samples_"
                f"{level:02d}{suffix}")
        img = np.zeros(outshape)
        img = io.add_vecs_to_img(vm, img, vecs, channels=level+1, grow=1)
        io.array2hdr(img, outf)

    def _linear(self, x, x1, x2):
        if len(self.levels) <= 2:
            return (x1, x2)[x]
        else:
            return (x2 - x1)/len(self.levels)*x + x1

    def threshold(self, idx):
        """threshold for determining sample count"""
        return self.accuracy * self._linear(idx, self.t0, self.t1)

    detailfunc = 'wav3'

    filters = {'prewitt': (np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])/3,
                           np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])/3),
               'sobel': (np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])/4,
                         np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])/3),
               'sobelswap': (np.array([[1, 2, -1], [0, 0, 0], [1, -2, -1]])/4,
                             np.array([[1, 0, 1], [-2, 0, 2], [-1, 0, -1]])/4),
               'cross': (np.array([[1, 0], [0, -1]])/2,
                         np.array([[0, 1], [-1, 0]])/2),
               'point': (np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])/3,
                         np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])),
               'wav': (np.array([[-1, 0, 0], [-1, 4, -1], [0, 0, -1]])/3,
                       np.array([[0, 0, -1], [-1, 4, -1], [-1, 0, 0]])/3),
               'wav3': (1 / 2 * np.array([[0, 0, 0], [-1, 2, -1], [0, 0, 0]]),
                        1 / 2 * np.array([[0, -1, 0], [0, 2, 0], [0, -1, 0]]),
                        1 / 2 * np.array([[-1, 0, 0], [0, 2, 0], [0, 0, -1]])),
               }

    def draw(self, level):
        """draw samples based on detail calculated from weights
        detail is calculated across direction only as it is the most precise
        dimension

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
                p = draw.get_detail(self.weights,
                                    *self.filters[self.detailfunc])
            # draw on pdf
            pdraws = draw.from_pdf(p, self.threshold(level),
                                   lb=self.lb, ub=self.ub)
        return pdraws, p

    def update_weights(self, si, lum):
        """update self.weights (which holds values used to calculate pdf)

        Parameters
        ----------
        si: np.array
            multidimensional indices to update
        lum:
            values to update with

        """
        self.weights[tuple(si)] = lum

    def run_callback(self, vecfs, name, point, posidx, vm):
        """handle class specific cleanup and lightpointKD construction"""
        outf = f'{self.scene.outdir}/{name}_{self.stype}_vals.out'
        vecs = []
        for vecf in vecfs:
            fsrc = open(vecf, 'rb')
            vecs.append(io.bytefile2np(fsrc, (-1, 6)))
            fsrc.close()
        vecs = np.concatenate(vecs)
        lightpoint = LightPointKD(self.scene, vecs, outf, src=self.stype,
                                  pt=point, write=True, srcn=self.srcn,
                                  posidx=posidx, vm=vm)
        [os.remove(vecf) for vecf in vecfs]
        return lightpoint

    def _dump_vecs(self, vecs, vecf):
        io.np2bytefile(vecs, vecf)

    def run(self, point, posidx, vm=None, plotp=False, log=False, outf=True,
            **kwargs):
        """

        Parameters
        ----------
        point: np.array
            point to sample
        posidx: int
            position index
        vm: raytraverse.mapper.ViewMapper
            view direction to sample
        plotp:
            plot weights, detail and vectors for each level
        log:
            whether to log level sampling rates
            can be 'scene', 'err' or None
            'scene' - logs to Scene log file
            'err' - logs to stderr
            anything else - does not log incremental progress
        outf: bool, optional
            some inheriting classes do not need an outfile, but this should
            not be changed unless the class stores the lum results directly
            and the run_callback of the class does not expect the file to exist.

        Returns
        -------

        """
        detaillog = True
        logerr = False
        if log == 'scene':
            logerr = False
        elif log == 'err':
            logerr = True
        else:
            detaillog = False
        if vm is None:
            vm = ViewMapper()
        point = np.asarray(point).flatten()[0:3]
        allc = 0
        name = f"{vm.name}_{posidx:06d}"
        if outf:
            outf = f'{self.scene.outdir}/{name}_{self.stype}_vals.out'
            f = open(outf, 'wb')
            f.close()
        else:
            outf = None
        self.scene.log(self, f"Started sampling {name} {self.stype}", logerr)
        self.scene.log(self, f"Settings: {self.engine_args[0]}", logerr)
        if detaillog:
            hdr = ['level ', '      shape', 'samples', '   rate',
                   'filesize (MB)']
            self.scene.log(self, '\t'.join(hdr), logerr)
        fsize = 0
        self.levels = vm.aspect
        # reset weights and engine args
        self.weights = np.full(self.levels[0], 1e-7, dtype=np.float32)
        self.engine.initialize(*self.engine_args, **self.engine_kwargs)
        vecfs = []
        for i in range(self.levels.shape[0]):
            shape = self.levels[i]
            self.weights = translate.resample(self.weights, shape)
            draws, p = self.draw(i)
            if len(draws) > 0:
                si, uv = self.sample_to_uv(draws, shape)
                xyz = vm.uv2xyz(uv)
                vecs = np.hstack((np.broadcast_to(point, xyz.shape), xyz))
                srate = si.shape[1]/np.prod(shape)
                fsize += 4*self.bands*self.srcn*si.shape[1]/1000000
                if detaillog:
                    row = (f"{i + 1} of {self.levels.shape[0]}\t"
                           f"{str(shape): >11}\t{si.shape[1]: >7}\t"
                           f"{srate: >7.02%}\t{fsize:.03f}")
                    self.scene.log(self, row, logerr)
                vecf = (f'{self.scene.outdir}/{name}_{self.stype}_vecs_'
                        f'{i:02d}.out')
                self._dump_vecs(vecs, vecf)
                vecfs.append(vecf)
                if plotp:
                    self._plot_p(p, i, vm, name)
                    self._plot_vecs(vecs[:, 3:], i, vm, name)
                lum = self.sample(vecf, vecs, outf)
                self.update_weights(si, lum)
                a = lum.shape[0]
                allc += a
        srate = allc/self.weights.size
        row = ['total sampling:', '- ', f"{allc: >7}", f"{srate: >7.02%}",
               f'{fsize:.03f}']
        self.scene.log(self, '\t'.join(row), logerr)
        return self.run_callback(vecfs, name, point, posidx, vm)
