# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import os
import re

import numpy as np

from raytraverse import translate, io, draw, renderer


class Sampler(object):
    """base sampling class

    Parameters
    ----------
    scene: raytraverse.scene.Scene
        scene class containing geometry, location and analysis plane
    engine: type, optional
        should inherit from raytraverse.renderer.Renderer
    fdres: int, optional
        final directional resolution given as log2(res)
    srcn: int, optional
        number of sources return per vector by run
    accuracy: float, optional
        parameter to set threshold at sampling level relative to final level
        threshold (smaller number will increase sampling, default is 1.0)
    idres: int, optional
        initial direction resolution (as log2(res))
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

    t0 = 2**-8
    t1 = .0625
    lb = .25
    ub = 8

    def __init__(self, scene, engine=renderer.Rtrace, fdres=9, srcn=1,
                 accuracy=1.0, idres=4,
                 stype='generic', srcdef=None, plotp=False,
                 bands=1, engine_args="", nproc=None, **kwargs):
        self.engine = engine()
        self._staticscene = True
        scene.log(self, "Initializing")
        #: int: number of spectral bands / channels returned by renderer
        #: based on given renderopts (user ensures these agree).
        self.bands = bands
        self.scene = scene
        #: func: mapper to use for sampling
        self.samplemap = self.scene.view
        #: int: number of sources return per vector by run
        self.srcn = srcn
        self.accuracy = accuracy
        #: int: initial direction resolution (as log2(res))
        self.idres = idres
        self.levels = fdres
        #: np.array: holds weights for self.draw
        self.weights = np.full(np.concatenate((self.scene.area.ptshape,
                                               self.levels[0])), 1e-7,
                               dtype=np.float32)
        #: int: index of next level to sample
        self.idx = 0
        #: str: sampler type
        self.stype = stype
        self.compiledscene = srcdef
        self.engine.initialize(engine_args, self.compiledscene, nproc=nproc,
                               iot="ff", **kwargs)
        self.plotp = plotp
        self.levelsamples = np.ones(self.levels.shape[0])
        # track vector files written for concatenation / cleanup after run
        self._vecfiles = []

    def __del__(self):
        self.scene.log(self, "Closed")
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
    def idx(self):
        """sampling level

        :getter: Returns the sampling level
        :setter: Set the sampling level and associated values (temp, shape)
        :type: int
        """
        return self._idx

    @idx.setter
    def idx(self, idx):
        self._idx = idx

    @property
    def levels(self):
        """sampling scheme

        :getter: Returns the sampling scheme
        :setter: Set the sampling scheme from (ptres, fdres, skres)
        :type: np.array
        """
        return self._levels

    @levels.setter
    def levels(self, fdres):
        """calculate sampling scheme"""
        self._levels = np.array([(2**i*self.scene.view.aspect, 2**i)
                                 for i in range(self.idres, fdres+1, 1)])

    @property
    def scene(self):
        """scene information

        :getter: Returns this sampler's scene
        :setter: Set this sampler's scene and create octree with source desc
        :type: raytraverse.scene.Scene
        """
        return self._scene

    @scene.setter
    def scene(self, scene):
        """Set this sampler's scene and create sky octree"""
        self._scene = scene

    def sample(self, vecf, vecs):
        """generic sample function

        Parameters
        ----------
        vecf: str
            path of file name with sample vectors
            shape (N, 6) vectors in binary float format
        vecs: np.array
            sample vectors (subclasses can choose which to use)

        Returns
        -------
        lum: np.array
            array of shape (N,) to update weights
        """
        outf = f'{self.scene.outdir}/{self.stype}_vals.out'
        f = open(outf, 'a+b')
        lumb = self.engine.call(vecf, outf=f)
        f.close()
        shape = (-1, self.srcn, self.bands)
        lum = io.bytes2np(lumb, shape)
        return np.squeeze(lum, axis=-1)

    def _uv2xyz(self, uv, si):
        """including to allow overriding mapping bevahior of daughter classes"""
        return self.samplemap.uv2xyz(uv)

    def _offset(self, shape):
        """for modifying jitter behavior of UV direction samples

        Parameters
        ----------
        shape: tuple
            shape of samples to jitter/offset
        """
        return np.random.default_rng().random(shape)/self.levels[self.idx][-1]

    def sample_idx(self, pdraws):
        """generate samples vectors from flat draw indices

        Parameters
        ----------
        pdraws: np.array
            flat index positions of samples to generate

        Returns
        -------
        si: np.array
            index array of draws matching samps.shape
        vecs: np.array
            sample vectors
        """
        shape = np.concatenate((self.scene.area.ptshape, self.levels[self.idx]))
        # index assignment
        si = np.stack(np.unravel_index(pdraws, shape))
        # convert to UV directions and positions
        uv = si.T[:, -2:]/shape[3]
        pos = self.scene.area.uv2pt((si.T[:, 0:2] + .5)/shape[0:2])
        uv += self._offset(uv.shape)
        if pos.size == 0:
            xyz = pos
        else:
            xyz = self._uv2xyz(uv, si)
        vecs = np.hstack((pos, xyz))
        return si, vecs

    def dump_vecs(self, vecs, si=None):
        """save vectors to file

        Parameters
        ----------
        vecs: np.array
            ray directions to write
        si: np.array, optional
            sample indices
        """
        outf = f'{self.scene.outdir}/{self.stype}_vecs_{self.idx:02d}.out'
        f = open(outf, 'wb')
        f.write(io.np2bytes(vecs))
        f.close()
        if si is not None:
            ptidx = np.ravel_multi_index((si[0], si[1]), self.scene.area.ptshape)
            outif = f'{self.scene.outdir}/{self.stype}_vecpidx_{self.idx:02d}.out'
            f = open(outif, 'wb')
            f.write(io.np2bytes(ptidx))
            f.close()
        else:
            outif = None
        self._vecfiles.append((outif, outf))
        return outf

    def _plot_p(self, p, suffix=".hdr", fisheye=True):
        ps = p.reshape(self.weights.shape)
        outshape = (1024, 512)
        res = outshape[-1]
        if fisheye:
            pixelxyz = self.samplemap.pixelrays(res)
            uv = self.samplemap.xyz2uv(pixelxyz.reshape(-1, 3))
            pdirs = np.concatenate((pixelxyz[0:res], -pixelxyz[res:]), 0)
            mask = self.samplemap.in_view(pdirs, indices=False).reshape(outshape)
            for i, ws in enumerate(self.weights):
                for j, w in enumerate(ws):
                    ij = translate.uv2ij(uv, w.shape[-1])
                    ptidx = np.ravel_multi_index((i, j), self.scene.area.ptshape)
                    outw = (f"{self.scene.outdir}_{self.stype}_weights_"
                            f"{ptidx:04d}_{self.idx+1:02d}{suffix}")
                    outp = (f"{self.scene.outdir}_{self.stype}_detail_"
                            f"{ptidx:04d}_{self.idx+1:02d}{suffix}")
                    img = w[ij[:, 0], ij[:, 1]].reshape(outshape)
                    io.array2hdr(np.where(mask, img, 0), outw)
                    img = ps[i, j][ij[:, 0], ij[:, 1]].reshape(outshape)
                    io.array2hdr(np.where(mask, img, 0), outp)
        else:
            for i, ws in enumerate(self.weights):
                for j, w in enumerate(ws):
                    ptidx = np.ravel_multi_index((i, j), self.scene.area.ptshape)
                    outw = (f"{self.scene.outdir}_{self.stype}_weights_"
                            f"{ptidx:04d}_{self.idx+1:02d}{suffix}")
                    outp = (f"{self.scene.outdir}_{self.stype}_detail_"
                            f"{ptidx:04d}_{self.idx+1:02d}{suffix}")
                    io.array2hdr(translate.resample(w[-1::-1], outshape,
                                                    radius=0), outw)
                    io.array2hdr(translate.resample(ps[i, j][-1::-1], outshape,
                                                    radius=0), outp)

    def _plot_vecs(self, idx, vecs, level=0):
        vm = self.samplemap
        outshape = (1024, 512)
        for i, ws in enumerate(self.weights):
            for j, w in enumerate(ws):
                ptidx = np.ravel_multi_index((i, j), self.scene.area.ptshape)
                img = np.zeros(outshape)
                img = io.add_vecs_to_img(vm, img, vecs[np.equal(idx, ptidx)],
                                         channels=level)
                outf = (f"{self.scene.outdir}_{self.stype}_samples_"
                        f"{ptidx:04d}_{level:02d}.hdr")
                io.array2hdr(img, outf)

    def _linear(self, x, x1, x2):
        if len(self.levels) <= 2:
            return (x1, x2)[x]
        else:
            return (x2 - x1)/len(self.levels)*x + x1

    def threshold(self, idx):
        """threshold for determining sample count"""
        return self.accuracy * self._linear(idx, self.t0, self.t1)

    detailfunc = 'wavelet'

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
               }

    def draw(self):
        """draw samples based on detail calculated from weights
        detail is calculated across direction only as it is the most precise
        dimension

        Returns
        -------
        pdraws: np.array
            index array of flattened samples chosen to sample at next level
        """
        dres = self.levels[self.idx]
        pres = self.scene.area.ptshape
        # sample all if weights is not set or all even
        if self.idx == 0 and np.var(self.weights) < 1e-9:
            pdraws = np.arange(np.prod(dres)*np.prod(pres))
            p = np.ones(self.weights.shape)
        else:
            # use weights directly on first pass
            if self.idx == 0:
                p = self.weights.ravel()
            # use wavelet transform
            elif self.detailfunc == 'wavelet':
                daxes = (len(pres) + len(dres) - 2, len(pres) + len(dres) - 1)
                p = draw.get_detail(self.weights, daxes)
            # use filter banks
            else:
                p = draw.get_detail_filter(self.weights,
                                           *self.filters[self.detailfunc])
            # draw on pdf
            pdraws = draw.from_pdf(p, self.threshold(self.idx),
                                   lb=self.lb, ub=self.ub)
        if self.plotp:
            self._plot_p(p, fisheye=True)
        return pdraws

    def update_pdf(self, si, lum):
        """update self.weights (which holds values used to calculate pdf)

        Parameters
        ----------
        si: np.array
            multidimensional indices to update
        lum:
            values to update with

        """
        self.weights[tuple(si)] = lum

    def run_callback(self):
        outf = f'{self.scene.outdir}/{self.stype}_vecs.out'
        # output traced rays for performance comparison
        # outft = f'{self.scene.outdir}/{self.stype}_vecs_trace.out'
        # f = open(outft, 'wb')
        # for idxf, vecf in self._vecfiles:
        #     fsrc = open(vecf, 'rb').read()
        #     f.write(fsrc)
        # f.close()
        f = open(outf, 'wb')
        for idxf, vecf in self._vecfiles:
            fsrc = open(vecf, 'rb')
            vecs = io.bytefile2np(fsrc, (-1, 6))
            fsrc.close()
            os.remove(vecf)
            if idxf is not None:
                fidx = open(idxf, 'rb')
                ptidx = io.bytefile2np(fidx, (1, -1))
                fidx.close()
                os.remove(idxf)
                dvecs = vecs[:, 3:]
                indexed_vecs = np.vstack((ptidx, dvecs.T)).T
                f.write(io.np2bytes(indexed_vecs))
                if self.plotp:
                    level = int(re.split(r"[_.]", idxf)[-2]) + 1
                    self._plot_vecs(ptidx.ravel(), dvecs, level)

            else:
                f.write(io.np2bytes(vecs))
        f.close()

    def get_scheme(self):
        scheme = np.ones((self.levels.shape[0], self.levels.shape[1] + 4))
        scheme[:, 2:-2] = self.levels
        scheme[:, 0:2] = self.scene.area.ptshape
        scheme[:, -2] = self.srcn
        scheme[:, -1] = self.levelsamples
        return scheme.astype(int)

    # @profile
    def run(self):
        """execute sampler"""
        allc = 0
        f = open(f'{self.scene.outdir}/{self.stype}_vals.out', 'wb')
        f.close()
        self.scene.log(self, f"Started sampling {self.stype}")
        hdr = ['level', 'shape', 'samples', 'rate', 'filesize (MB)']
        self.scene.log(self, '\t'.join(hdr))
        fsize = 0
        for i in range(self.idx, self.levels.shape[0]):
            shape = np.concatenate((self.scene.area.ptshape, self.levels[i]))
            self.idx = i
            self.weights = translate.resample(self.weights, shape)
            draws = self.draw()
            if draws is None:
                srate = 0.0
                row = [f'{i + 1} of {self.levels.shape[0]}', str(shape),
                       '0', f"{srate:.02%}", f'{fsize:.03f}']
                self.scene.log(self, '\t'.join(row))
            else:
                self.levelsamples[self.idx] = draws.size
                si, vecs = self.sample_idx(draws)
                srate = si.shape[1]/np.prod(shape)
                fsize += 4*self.bands*self.srcn*si.shape[1]/1000000
                row = [f'{i+1} of {self.levels.shape[0]}', str(shape),
                       str(si.shape[1]), f"{srate:.02%}", f'{fsize:.03f}']
                self.scene.log(self, '\t'.join(row))
                vecf = self.dump_vecs(vecs, si)
                lum = self.sample(vecf, vecs)
                self.update_pdf(si, lum)
                a = lum.shape[0]
                allc += a
        srate = allc/self.weights.size
        row = ['total sampling:', '-', str(allc), f"{srate:.02%}", f'{fsize:.03f}']
        self.scene.log(self, '\t'.join(row))
        self.run_callback()
