# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import clasp.script_tools as cst
from raytraverse import translate, io, wavelet, quickplot
from clipt import mplt
from memory_profiler import profile


class Sampler(object):
    """parent sampling class

    overload sample() to implement

    Parameters
    ----------
    scene: raytraverse.scene.Scene
        scene class containing geometry, location and analysis plane
    fdres: int, optional
        final directional resolution given as log2(res)
    srcn: int, optional
        number of sources return per vector by run
    t0: float, optional
        in range 0-1, fraction of uniform random samples taken at first step
    t1: float, optional
        in range 0-t0, fraction of uniform random samples taken at final step
    minrate: float, optional
        in range 0-1, fraction of samples at first step.
    minrate: float, optional
        in range 0-1, fraction of samples at final step (this is not the total
        sampling rate, which depends on the number of levels).
    idres: int, optional
        initial direction resolution (as log2(res))
    stype: str, optional
        sampler type (prefixes output files)
    append: bool, optional
        append results of run to existing sample files (if present in scene)
        otherwise overwrites with call to run
    """

    def __init__(self, scene, fdres=9, srcn=1, t0=.1, t1=.01, maxrate=1.0,
                 minrate=.05, idres=4, stype='generic', append=False,
                 srcdef=None, plotp=False, **kwargs):
        self.scene = scene
        #: func: mapper to use for sampling
        self.samplemap = self.scene.view
        #: int: number of sources return per vector by run
        self.srcn = srcn
        #: int: initial direction resolution (as log2(res))
        self.idres = idres
        self.levels = fdres
        #: float: fraction of uniform random samples taken at first step
        self.t0 = t0
        #: float: fraction of uniform random samples taken at final step
        self.t1 = t1
        #: float: fraction of samples at first step
        self.maxrate = maxrate
        #: float: fraction of samples at final step
        self.minrate = minrate
        #: np.array: holds weights for self.draw
        self.weights = np.full(np.concatenate((self.scene.ptshape,
                                               self.levels[0])), 1e-7)
        #: int: index of next level to sample
        self.idx = 0
        #: str: sampler type
        self.stype = stype
        # bool: overwrites existing results files when false
        self.append = append
        self.compiledscene = srcdef
        self.plotp = plotp

    def __del__(self):
        try:
            os.remove(self.compiledscene)
        except (IOError, TypeError):
            pass

    @property
    def compiledscene(self):
        return self._compiledscene

    @compiledscene.setter
    def compiledscene(self, src):
        self._compiledscene = f'{self.scene.outdir}/{self.stype}.oct'
        if src is None:
            pass
        else:
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
        try:
            x = self.idx/(self.levels.shape[0]-1)
        except ZeroDivisionError:
            x = 1
        self.set_rates(x)

    def set_rates(self, x):
        self._sample_t = wavelet.get_uniform_rate(x, self.t0, self.t1)
        self._sample_rate = wavelet.get_sample_rate(x, self.minrate,
                                                    self.maxrate)

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
        :setter: Set this sampler's scene and create sky octree
        :type: raytraverse.scene.Scene
        """
        return self._scene

    @scene.setter
    def scene(self, scene):
        """Set this sampler's scene and create sky octree"""
        self._scene = scene

    def sample(self, vecs, call=None, **kwargs):
        """dummy sample function
        """
        if call is None:
            raise NotImplementedError(f'{self.__class__} does'
                                      ' not have a valid sample method')
        outf = f'{self.scene.outdir}/{self.stype}_vals.out'
        lum = io.call_sampler(outf, call, vecs)
        return lum

    def _uv2xyz(self, uv, si):
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
        shape = np.concatenate((self.scene.ptshape, self.levels[self.idx]))
        # index assignment
        si = np.stack(np.unravel_index(pdraws, shape))
        # convert to UV directions and positions
        uv = si.T[:, -2:]/shape[3]
        pos = self.scene.area.uv2pt((si.T[:, 0:2] + .5)/shape[0:2])
        uv += (np.random.default_rng().random(uv.shape))/shape[3]
        # mplt.quick_scatter([uv[:, 0]], [uv[:, 1]], ms=3, lw=0)
        xyz = self._uv2xyz(uv, si)
        vecs = np.hstack((pos, xyz))
        return si, vecs

    def dump_vecs(self, si, vecs):
        """save vectors to file

        Parameters
        ----------
        si: np.array
            sample indices
        vecs: np.array
            ray directions to write
        """
        ptidx = np.ravel_multi_index((si[0], si[1]), self.scene.ptshape)
        outf = f'{self.scene.outdir}/{self.stype}_vecs.out'
        f = open(outf, 'ab')
        f.write(io.np2bytes(np.vstack((ptidx.reshape(1, -1), vecs.T)).T))
        f.close()

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
        pres = self.scene.ptshape
        if self._sample_rate == 1:
            pdraws = np.arange(np.prod(dres)*np.prod(pres))
        else:
            # direction detail
            daxes = (len(pres) + len(dres) - 2, len(pres) + len(dres) - 1)
            p = wavelet.get_detail(self.weights, daxes)

            # q = np.quantile(p, 1-self._sample_rate)
            self._sample_rate = np.sum(p > 0.01 * 2**(2 * self.idx - 1) /
                                       self.levels[-1, -1])/p.size
            p = p*(1 - self._sample_t) + np.median(p)*self._sample_t
            if self.plotp:
                quickplot.imshow(np.log10(p.reshape(self.weights.shape)[0, 0]),
                                 [20, 10])
            # draw on pdf
            nsampc = int(self._sample_rate*self.weights.size)
            pdraws = np.random.default_rng().choice(p.size, nsampc,
                                                    replace=False,
                                                    p=p/np.sum(p))
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
        pass

    def get_scheme(self):
        scheme = np.ones((self.levels.shape[0], self.levels.shape[1] + 4))
        scheme[:, 2:-2] = self.levels
        scheme[:, 0:2] = self.scene.ptshape
        scheme[:, -2] = self.srcn
        for i in range(scheme.shape[0]):
            x = i/(self.levels.shape[0] - 1)
            a = wavelet.get_sample_rate(x, self.minrate, self.maxrate)
            scheme[i, -1] = a*np.product(scheme[i])
        return scheme.astype(int)

    def print_sample_cnt(self):
        an = 0
        print("shape\tguided\tuniform\ttotal")
        for i, l in enumerate(self.levels):
            shape = np.concatenate((self.scene.ptshape, self.levels[i]))
            x = i/(self.levels.shape[0] - 1)
            a = int(wavelet.get_sample_rate(x, self.minrate,
                                            self.maxrate)*np.product(shape))
            t = wavelet.get_uniform_rate(x, self.t0, self.t1)
            an += a
            print(f"{l}\t{int(a*(1-t))}\t{int(a*t)}\t{a}")
        print("total", an)

    # @profile
    def run(self, **skwargs):
        """execute sampler

        Parameters
        ----------
        skwargs
            keyword arguments passed to self.sample

        """

        allc = 0
        if not self.append:
            f = open(f'{self.scene.outdir}/{self.stype}_vecs.out', 'wb')
            f.close()
            f = open(f'{self.scene.outdir}/{self.stype}_vals.out', 'wb')
            f.close()
        print('Sampling...')
        hdr = ['level', 'shape', 'samples', 'rate', 'filesize (MB)']
        print('{:>8}  {:>25}  {:<10}  {:<8}  {}'.format(*hdr))
        fsize = 0
        for i in range(self.idx, self.levels.shape[0]):
            shape = np.concatenate((self.scene.ptshape, self.levels[i]))
            self.idx = i
            self.weights = translate.resample(self.weights, shape)
            draws = self.draw()
            si, vecs = self.sample_idx(draws)
            srate = si.shape[1]/np.prod(shape)
            fsize += 12*self.srcn*si.shape[1]/1000000
            row = [f'{i+1} of {self.levels.shape[0]}', str(shape), si.shape[1],
                   f"{srate:.02%}", fsize]
            print('{:>8}  {:>25}  {:<10}  {:<8}  {:.03f}'.format(*row))
            self.dump_vecs(si, vecs[:, 3:])
            lum = self.sample(vecs, **skwargs)
            self.update_pdf(si, lum)
            a = lum.shape[0]
            allc += a
        print("-"*70)
        srate = allc/self.weights.size
        row = ['total sampling:', allc, f"{srate:.02%}", fsize]
        print('{:<35}  {:<10}  {:<8}  {:.03f}'.format(*row))
        self.run_callback()