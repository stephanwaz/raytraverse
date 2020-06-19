# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from raytraverse import translate, io, wavelet
from memory_profiler import profile

scbinscal = ("""
{ map U/V axis to bin divisions }
axis(x) : mod(floor(side * x), side);

{ get bin of u,v }
binl(u, v) : axis(u)*side + axis(v);

{ shirley-chiu disk to square (with spherical term) }
pi4 : PI/4;
n = if(Dz, 1, -1);
r2 = 1 - n*Dz;
x = Dx/sqrt(2 - r2);
y = Dy/sqrt(2 - r2);
r = sqrt( sq(x) + sq(y));
ph = atan2(x, y);
phi = ph + if(-pi4 - ph, 2*PI, 0);
a = if(pi4 - phi, r, if(3*pi4 - phi, -(phi - PI/2)*r/pi4, if(5*pi4 - phi,"""
             """ -r, (phi - 3*PI/2)*r/pi4)));
b = if(pi4 - phi, phi*r/pi4, if(3*pi4 - phi, r, if(5*pi4 - phi, """
             """-(phi - PI)*r/pi4, -r)));

{ map to (0,2),(0,1) matches raytraverse.translate.xyz2uv}
U = (a*n + if(n, 1, 3))/2;
V = (b + 1)/2;

bin = binl(V, U);
""")

scxyzcal = """
x1 = .5;
x2 = .5;

U = ((bin - mod(bin, side)) / side + x1)/side;
V = (mod(bin, side) + x2)/side;

n = if(U - 1, -1, 1);
ur = if(U - 1, U - 1, U);
a = 2 * ur - 1;
b = 2 * V - 1;
conda = sq(a) - sq(b);
condb = abs(b) - FTINY;
r = if(conda, a, if(condb, b, 0));
phi = if(conda, b/(2*a), if(condb, 1 - a/(2*b), 0)) * PI/2;
sphterm = r * sqrt(2 - sq(r));
Dx = n * cos(phi)*sphterm;
Dy = sin(phi)*sphterm;
Dz = n * (1 - sq(r));
"""


class Sampler(object):
    """parent sampling class (returns random samples with value == 1)

    overload draw() and sample() to implement

    Parameters
    ----------
    scene: raytraverse.scene.Scene
        scene class containing geometry, location and analysis plane
    dndepth: int, optional
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
    """

    def __init__(self, scene, dndepth=9, srcn=1, t0=.1, t1=.01, maxrate=1.0,
                 minrate=.05, idres=4, stype='generic'):
        self.scene = scene
        #: int: number of sources return per vector by run
        self.srcn = srcn
        #: int: initial direction resolution (as log2(res))
        self.idres = idres
        self.levels = dndepth
        #: float: fraction of uniform random samples taken at first step
        self.t0 = t0
        #: float: fraction of uniform random samples taken at final step
        self.t1 = t1
        #: float: fraction of samples at first step
        self.maxrate = maxrate
        #: float: fraction of samples at final step
        self.minrate = minrate
        #: int: index of next level to sample
        self.idx = 0
        #: str: sampler type
        self.stype = stype
        #: np.array: holds weights for self.draw
        self.weights = np.full(np.concatenate((self.scene.ptshape,
                                               self.levels[0])), 1e-7)

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
        :setter: Set the sampling scheme from (ptres, dndepth, skres)
        :type: np.array
        """
        return self._levels

    @levels.setter
    def levels(self, dndepth):
        """calculate sampling scheme"""
        self._levels = np.array([(2**i*self.scene.view.aspect, 2**i)
                                 for i in range(self.idres, dndepth+1, 1)])

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

    def sample(self, vecs, **kwargs):
        """dummy sample function
        """
        lum = np.ones((vecs.shape[0], self.srcn))
        print(lum.shape)
        return lum

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
        uv = si.T[:, 2:]/shape[3]
        pos = self.scene.area.uv2pt((si.T[:, 0:2] + .5)/shape[0:2])

        uv += (np.random.default_rng().random(uv.shape))/shape[3]
        # uv += .5/shape[3]
        xyz = self.scene.view.uv2xyz(uv)
        # xyz = translate.uv2xyz(uv, axes=(0, 2, 1))
        vecs = np.hstack((pos, xyz))
        return si, vecs

    def dump(self, ptidx, vecs, vals, wait=False):
        """save values to file
        see io.write_npy

        Parameters
        ----------
        ptidx: np.array
            point indices
        vecs: np.array
            ray directions to write
        vals: np.array
            values to write
        wait: bool, optional
            if false, does not wait for files to write before returning

        Returns
        -------
        None
        """
        outf = f'{self.scene.outdir}/{self.stype}_{self.idx}'
        if wait:
            return io.write_npy(ptidx, vecs, vals, outf)
        else:
            executor = ThreadPoolExecutor()
            return executor.submit(io.write_npy, ptidx, vecs, vals, outf)

    def draw(self):
        """define pdf generated from samps and draw next set of samples

        Returns
        -------
        pdraws: np.array
            index array of flattened samps chosed to sample at next level
        """
        p = np.ones(self.weights.size)
        # draw on pdf
        nsampc = int(self._sample_rate*self.weights.size)
        pdraws = np.random.default_rng().choice(p.size, nsampc, replace=False, p=p/np.sum(p))
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
        self.weights[tuple(si)] = np.max(lum, 1)

    def save_pdf(self):
        outf = f'{self.scene.outdir}/{self.stype}_pdf'
        np.save(outf, self.weights)

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

        dumps = []
        for i in range(self.idx, self.levels.shape[0]):
            shape = np.concatenate((self.scene.ptshape, self.levels[i]))
            self.idx = i
            self.weights = translate.resample(self.weights, shape)
            draws = self.draw()
            si, vecs = self.sample_idx(draws)
            ptidx = np.ravel_multi_index((si[0], si[1]), self.scene.ptshape)
            srate = si.shape[1]/np.prod(shape)
            print(f"{shape} sampling: {si.shape[1]}\t{srate:.02%}")
            lum = self.sample(vecs, **skwargs)
            dumps.append(self.dump(ptidx, vecs[:, 3:], lum))
            self.update_pdf(si, lum)
            a = lum.shape[0]
            allc += a
        print("--------------------------------------")
        srate = allc/self.weights.size
        print(f"asamp: {allc}\t{srate:.02%}")
        self.save_pdf()
        for dump in dumps:
            if dump is None:
                pass
            else:
                wait = dump.result()


