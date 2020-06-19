# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

import os
import shlex
from subprocess import Popen, PIPE

import numpy as np

from clasp import script_tools as cst
from scipy.ndimage import maximum_filter

from raytraverse import optic, io, wavelet, Sampler, translate

from memory_profiler import profile


class SunSampler(Sampler):
    """sample contributions from direct suns.

    Parameters
    ----------
    scene: raytraverse.scene.Scene
        scene class containing geometry, location and analysis plane
    sunres: np.array, path or None, optional
        np.array (or path to saved np.array) containing sky contribution values
        to use as probabilty for drawing suns and determining number of suns
        to calculate, if none, suns are drawn uniformly from sun path of scene
        and the sunres determines the number.
    sunres: float, optional
        minimum average seperation between sources (twice desired error)
    srct: float, optional
        threshold of sky contribution for determining appropriate srcn
    """

    def __init__(self, scene, sunres=5.0, srct=.01, maxspec=.3,
                 dndepth=12, t0=.05, t1=.01, maxrate=.01, minrate=.005, idres=10,
                 **kwargs):
        self.ndsamps = 0
        super(SunSampler, self).__init__(scene, stype='sun', dndepth=dndepth,
                                         t0=t0, t1=t1, minrate=minrate,
                                         maxrate=maxrate, idres=idres, **kwargs)
        self.srct = srct
        if sunres < .61:
            print('Warning! minimum sunres is .65 to avoid overlap and allow')
            print('for jittering position, sunres set to .65')
            sunres = .65
        self.suns = sunres
        self.maxspec = maxspec
        self.sweights = 0
        self.init_weights()

    def init_weights(self):
        shape = np.concatenate((self.scene.ptshape, self.levels[self.idx]))
        skypdf = f"{self.scene.outdir}/sky_pdf.npy"
        try:
            skypdf = np.load(skypdf)
        except FileNotFoundError:
            print('Warning! sunsampler initialized without vector weights')
            self._weights = np.full(shape, 1e-7)
            vis = np.ones(shape)
        else:
            vis = translate.resample(skypdf, shape)
            skypdf[skypdf > self.maxspec] = 0
            self.weights = translate.resample(skypdf*.25, shape)
        self.init_sweights(vis)

    def init_sweights(self, vis):
        """at resolution 512 a 5x5 square will enclose and box in the solar
        disk"""
        sh0 = np.concatenate((self.scene.ptshape,
                             (1024*self.scene.view.aspect, 1024)))
        sh1 = np.concatenate((self.scene.ptshape, self.levels[self.idx]))
        vis = translate.resample(vis, sh0)
        sunuv = self.scene.view.xyz2uv(self.suns)
        inview = self.scene.in_view(sunuv)
        suni = translate.uv2ij(sunuv[inview], 1024).T
        suns = np.zeros(sh0)
        s = np.s_[..., suni[0], suni[1]]
        isviz = vis[s] > self.srct/2
        suns[s] = isviz
        suns = maximum_filter(suns, (1, 1, 7, 7), mode='wrap')
        self.sweights = translate.resample(suns, sh1, gauss=False)
        self.ndsamps = np.sum(self.sweights > 0)

    @property
    def suns(self):
        """holds pdf for sampling suns

        :getter: Returns the sun source array
        :setter: Set the sun source array and write to files
        :type: np.array
        """
        return self._suns

    @suns.setter
    def suns(self, sunres):
        """set the skydetail array and determine sample count and spacing"""
        sunfile = f"{self.scene.outdir}/suns.rad"
        if os.path.isfile(sunfile):
            self.load_suns(sunfile)
        else:
            uvsize = int(np.floor(90/sunres)*2)
            skyb = self.load_sky_facs(uvsize)
            si = np.stack(np.unravel_index(np.arange(skyb.size), skyb.shape))
            uv = si.T/uvsize
            ib = self.scene.in_solarbounds(uv + .5/uvsize,
                                           size=1/uvsize).reshape(skyb.shape)
            suncount = np.sum(skyb*ib > self.srct)
            print(suncount)
            skyd = wavelet.get_detail(skyb, (0, 1)).reshape(skyb.shape)
            sb = (skyb + skyd)/np.min(skyb + skyd)
            sd = (sb * ib).ravel()
            sdraws = np.random.default_rng().choice(skyb.size, suncount, replace=False,
                                      p=sd/np.sum(sd))
            si = np.stack(np.unravel_index(sdraws, skyb.shape))
            border = .3*uvsize/180
            j = np.random.default_rng().uniform(border, 1-border, si.T.shape)
            uv = (si.T + j)/uvsize
            self._suns = translate.uv2xyz(uv, xsign=1)
            self.write_suns(sunfile)
        self.srcn = self.suns.shape[0]

    def set_rates(self, x):
        super(SunSampler, self).set_rates(x)
        self._sample_drate = self.sun_sample_rate(x)

    def sun_sample_rate(self, x):
        return 1/(1 + (x > 0)*3)

    def load_suns(self, sunfile):
        """load suns from file

        void light solar00001 0 0 3 1 1 1
        solar00001 source sun00001 0 0 4 X Y Z 0.533
        .
        .
        .
        void light solarNNNNN 0 0 3 1 1 1
        solarNNNNN source sunNNNNN 0 0 4 X Y Z 0.533

        Parameters
        ----------
        sunfile
        """
        f = open(sunfile, 'r')
        sund = f.read().split('source')[1:]
        self._suns = np.array([s.split()[4:7] for s in sund]).astype(float)

    def write_suns(self, sunfile):
        """write suns to file and make sun octree

        Parameters
        ----------
        sunfile
        """
        sunoct = f'{self.scene.outdir}/{self.stype}.oct'
        f = open(sunfile, 'w')
        g = open(f'{self.scene.outdir}/{self.stype}_modlist.txt', 'w')
        for i, s in enumerate(self.suns):
            mod = f"solar{i:05d}"
            name = f"sun{i:05d}"
            d = f"{s[0]} {s[1]} {s[2]}"
            print(mod, file=g)
            print(f"void light {mod} 0 0 3 1 1 1", file=f)
            print(f"{mod} source {name} 0 0 4 {d} 0.533", file=f)
        f.close()
        g.close()
        f = open(sunoct, 'wb')
        cst.pipeline([f'oconv -f -i {self.scene.outdir}/scene.oct {sunfile}'],
                     outfile=f, close=True)

    def load_sky_facs(self, uvsize):
        try:
            skyb = np.load(f'{self.scene.outdir}/sky_skydetail.npy')
        except FileNotFoundError:
            print('Warning! sunsampler initialized without sky weights')
            skyb = np.ones((uvsize, uvsize))
        else:
            skyb = translate.resample(skyb, (uvsize, uvsize))
        return skyb

    def print_sample_cnt(self):
        super(SunSampler, self).print_sample_cnt()
        print(f'direct view samples:')
        for i in range(len(self.levels)):
            try:
                x = i/(self.levels.shape[0] - 1)
            except ZeroDivisionError:
                x = 1
            print(f"level {i}: {self.ndsamps * self.sun_sample_rate(x)}")

    def sample(self, vecs, rcopts='-ab 7 -ad 60000 -as 30000 -lw 1e-7',
               nproc=12, executable='rcontrib'):
        """call rendering engine to sample sky contribution

        Parameters
        ----------
        vecs: np.array
            shape (N, 6) vectors to calculate contributions for
        rcopts: str, optional
            option string to send to executable
        nproc: int, optional
            number of processes executable should use
        executable: str, optional
            rendering engine binary

        Returns
        -------
        lum: np.array
            array of shape (N, binnumber) with sun coefficients
        """
        fdr = self.scene.outdir
        octr = f"{fdr}/{self.stype}.oct"
        rc = (f"{executable} -fff {rcopts} -h -n {nproc} "
              f"-M {fdr}/{self.stype}_modlist.txt {octr}")
        p = Popen(shlex.split(rc), stdout=PIPE,
                  stdin=PIPE).communicate(io.np2bytes(vecs))
        lum = optic.rgb2rad(io.bytes2np(p[0], (-1, 3)))
        return lum.reshape(-1, self.srcn)

    # @profile
    def draw(self):
        """draw samples based on detail calculated from weights
        detail is calculated across direction only as it is the most precise
        dimension

        Returns
        -------
        pdraws: np.array
            index array of flattened samples chosen to sample at next level
        """
        self.sweights = translate.resample(self.sweights, self.weights.shape, radius=0)
        if self.idx > 0:
            dres = self.levels[self.idx]
            pres = self.scene.ptshape
            # direction detail
            daxes = tuple(range(len(pres), len(pres) + len(dres)))
            p = wavelet.get_detail(self.weights, daxes)
            d = wavelet.get_detail(self.sweights, daxes)
        else:
            p = self.weights.ravel()
            d = self.sweights.ravel()
        p = p*(1 - self._sample_t) + np.median(p)*self._sample_t
        nsampc = int(self._sample_rate*self.weights.size)
        nsampd = int(self._sample_drate*self.ndsamps)
        # draw on pdf
        pdraws = np.random.default_rng().choice(p.size, nsampc, replace=False,
                                  p=p/np.sum(p))
        # add additional samples for direct sun view rays
        ddraws = np.random.default_rng().choice(p.size, nsampd, replace=False,
                                  p=d/np.sum(d))
        return np.concatenate((pdraws, ddraws))

    def update_pdf(self, si, lum):
        """update self.weights (which holds values used to calculate pdf)

        Parameters
        ----------
        si: np.array
            multidimensional indices to update
        lum:
            values to update with
        """
        # lum[lum > self.maxspec] = 0
        self.weights[tuple(si[:, :self.ndsamps])] = np.max(lum[:self.ndsamps], 1)
        self.sweights[tuple(si[:, -self.ndsamps:])] = np.max(lum[-self.ndsamps:], 1)

