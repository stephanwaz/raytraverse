# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import os
import sys

import numpy as np

import clasp.script_tools as cst
from raytraverse import translate, io, draw, quickplot


class Sampler(object):
    """base sampling class

    To implement a proper-subclass an engine attribute must be set to a
    renderer instance prior to calling Sampler.__init__. Also, the method
    sample must be overriden to properly set up arguments for the renderer.call

    Parameters
    ----------
    scene: raytraverse.scene.Scene
        scene class containing geometry, location and analysis plane
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
    engine = None

    def __init__(self, scene, fdres=9, srcn=1, accuracy=1.0, idres=4,
                 stype='generic', srcdef=None, plotp=False,
                 bands=1, engine_args="", nproc=None, **kwargs):
        name = type(self).__name__
        self._staticscene = True
        if name == "Sampler":
            raise NotImplementedError("Sampler base class should not be "
                                      "instantiated directly")
        elif not all(hasattr(self.engine, atr) for atr
                     in ['initialize', 'call']):
            raise NotImplementedError(f"Subclass {name} of Sampler is ill"
                                      " defined, an engine attribute must"
                                      " be set to a valid renderer class"
                                      " before invoking Sampler.__init__")
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
                               iot="ff")
        self.plotp = plotp
        self.levelsamples = np.ones(self.levels.shape[0])
        # track vector files written for concatenation / cleanup after run
        self._vecfiles = []

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
            self._compiledscene = f'{self.scene.outdir}/scene.oct'
        else:
            self._compiledscene = f'{self.scene.outdir}/{self.stype}.oct'
            if os.path.isfile(src):
                ocom = f'oconv -f -i {self.scene.outdir}/scene.oct {src}'
                inp = None
            else:
                ocom = f'oconv -f -i {self.scene.outdir}/scene.oct -'
                inp = src
            f = open(self.compiledscene, 'wb')
            cst.pipeline([ocom], outfile=f, inp=inp, close=True)

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

    def sample(self, vecf):
        """generic sample function
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
        uv += (np.random.default_rng().random(uv.shape))/shape[3]
        # mplt.quick_scatter([uv[:, 0]], [uv[:, 1]], ms=3, lw=0)
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
        if self.idx == 0 and np.var(self.weights) < 1e-9:
            pdraws = np.arange(np.prod(dres)*np.prod(pres))
        else:
            # direction detail
            daxes = (len(pres) + len(dres) - 2, len(pres) + len(dres) - 1)
            p = draw.get_detail(self.weights, daxes)
            if self.plotp:
                quickplot.imshow(np.log10(p.reshape(self.weights.shape)[0, 0]),
                                 [20, 10])
            # draw on pdf
            # threshold is set to accurracy at final
            threshold = self.accuracy * 4**(self.idx - len(self.levels))
            pdraws = draw.from_pdf(p, threshold)
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
                f.write(io.np2bytes(np.vstack((ptidx.reshape(1, -1),
                                               vecs[:, 3:].T)).T))
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
        print('Sampling...', file=sys.stderr)
        hdr = ['level', 'shape', 'samples', 'rate', 'filesize (MB)']
        print('{:>8}  {:>25}  {:<10}  {:<8}  {}'.format(*hdr), file=sys.stderr)
        fsize = 0
        for i in range(self.idx, self.levels.shape[0]):
            shape = np.concatenate((self.scene.area.ptshape, self.levels[i]))
            self.idx = i
            self.weights = translate.resample(self.weights, shape)
            draws = self.draw()
            if draws is None:
                srate = 0.0
                row = [f'{i + 1} of {self.levels.shape[0]}', str(shape),
                       0, f"{srate:.02%}", fsize]
                print('{:>8}  {:>25}  {:<10}  {:<8}  {:.03f}'.format(*row),
                      file=sys.stderr)
            else:
                self.levelsamples[self.idx] = draws.size
                si, vecs = self.sample_idx(draws)
                srate = si.shape[1]/np.prod(shape)
                fsize += 4*self.bands*self.srcn*si.shape[1]/1000000
                row = [f'{i+1} of {self.levels.shape[0]}', str(shape),
                       si.shape[1], f"{srate:.02%}", fsize]
                print('{:>8}  {:>25}  {:<10}  {:<8}  {:.03f}'.format(*row),
                      file=sys.stderr)
                vecf = self.dump_vecs(vecs, si)
                lum = self.sample(vecf)
                self.update_pdf(si, lum)
                a = lum.shape[0]
                allc += a
        print("-"*70, file=sys.stderr)
        srate = allc/self.weights.size
        row = ['total sampling:', allc, f"{srate:.02%}", fsize]
        print('{:<35}  {:<10}  {:<8}  {:.03f}'.format(*row), file=sys.stderr)
        self.run_callback()
