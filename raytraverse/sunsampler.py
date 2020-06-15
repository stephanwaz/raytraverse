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
# from scipy.ndimage import

from clasp import script_tools as cst
from scipy.interpolate import RectBivariateSpline

from raytraverse import optic, io, wavelet, Sampler, translate
from raytraverse.sampler import scbinscal


def load_sky_facs(skyb, uvsize):
    if skyb is None:
        skyb = np.ones((uvsize, uvsize))
    elif not isinstance(skyb, np.ndarray):
        skyb = np.load(skyb)
    oldc = np.linspace(0, 1, skyb.shape[0])
    newc = np.linspace(0, 1, uvsize)
    f = RectBivariateSpline(oldc, oldc, skyb, kx=1, ky=1)
    return f(newc, newc)


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
        minimum average seperation between sources
    srct: float, optional
        threshold of sky contribution for determining appropriate srcn
    """

    def __init__(self, scene, skyb=None, sunres=5.0, srct=.01, **kwargs):
        super(SunSampler, self).__init__(scene, stype='sun',  **kwargs)
        self.suns = (skyb, sunres, srct)
        self.mk_sun_files()

    @property
    def suns(self):
        """holds pdf for importance sampling suns

        :getter: Returns the skydetail array
        :setter: Set the skydetail array
        :type: np.array
        """
        return self._suns

    @suns.setter
    def suns(self, skyparams):
        """set the skydetail array and determine sample count and spacing"""
        skyb, sunres, srct = skyparams
        uvsize = int(np.floor(90/sunres)*2)
        skyb = load_sky_facs(skyb, uvsize)
        si = np.stack(np.unravel_index(np.arange(skyb.size), skyb.shape))
        uv = si.T/uvsize
        ib0 = self.scene.in_solarbounds(uv)
        ib1 = self.scene.in_solarbounds(uv + np.array([[1, 0]])/uvsize)
        ib2 = self.scene.in_solarbounds(uv + np.array([[1, 1]])/uvsize)
        ib3 = self.scene.in_solarbounds(uv + np.array([[0, 1]])/uvsize)
        ib = (ib0*ib1*ib2*ib3).reshape(skyb.shape)
        suncount = np.sum(skyb*ib > srct)
        skyd = wavelet.get_detail(skyb, (0, 1)).reshape(skyb.shape)
        sb = (skyb + skyd)/np.min(skyb + skyd)
        sd = (sb * ib).flatten()
        sdraws = np.random.choice(skyb.size, suncount, replace=False,
                                  p=sd/np.sum(sd))
        si = np.stack(np.unravel_index(sdraws, skyb.shape))
        uv = (si.T + np.random.random(si.T.shape))/uvsize
        self._suns = translate.uv2xyz(uv)


    def mk_sun_files(self):
        skyoct = f'{self.scene.outdir}/{self.stype}.oct'
        if not os.path.isfile(skyoct):
            skydef = ("void light skyglow 0 0 3 1 1 1 skyglow source sky 0 0 4"
                      " 0 0 1 180")
            skydeg = ("void glow skyglow 0 0 4 1 1 1 0 skyglow source sky 0 0 4"
                      " 0 0 1 180")
            f = open(skyoct, 'wb')
            cst.pipeline([f'oconv -i {self.scene.outdir}/scene.oct -'],
                         inp=skydeg, outfile=f, close=True)
            f = open(f'{self.scene.outdir}/sky_pm.oct', 'wb')
            cst.pipeline([f'oconv -i {self.scene.outdir}/scene.oct -'],
                         inp=skydef, outfile=f, close=True)
        f = open(f'{self.scene.outdir}/scbins.cal', 'w')
        f.write(scbinscal)
        f.close()

    def sample(self, vecs, rcopts='-ab 7 -ad 60000 -as 30000 -lw 1e-7',
               nproc=12, executable='rcontrib_pm', bwidth=1000):
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
        bwidth: int, optional
            if using photon mapping, the bandwidth parameter

        Returns
        -------
        lum: np.array
            array of shape (N, binnumber) with sky coefficients
        """
        fdr = self.scene.outdir
        if self.skypmap:
            rcopts += f' -ab -1 -ap {fdr}/sky.gpm {bwidth}'
            octr = f"{fdr}/sky_pm.oct"
        else:
            octr = f"{fdr}/sky.oct"
        rc = (f"{executable} -V+ -fff {rcopts} -h -n {nproc} -e "
              f"'side:{self.skres}' -f {fdr}/scbins.cal -b bin -bn {self.srcn} "
              f"-m skyglow {octr}")
        p = Popen(shlex.split(rc), stdout=PIPE,
                  stdin=PIPE).communicate(io.np2bytes(vecs))
        lum = optic.rgb2rad(io.bytes2np(p[0], (-1, 3)))
        return lum.reshape(-1, self.srcn)

    def draw(self, samps):
        """draw samples based on detail calculated from samps
        detail is calculated across position and direction seperately and
        combined by product (would be summed otherwise) to avoid drowning out
        the signal in the more precise dimensions (assuming a mismatch in step
        size and final stopping criteria

        Parameters
        ----------
        samps: np.array
            shape self.scene.ptshape + self.levels[self.idx]

        Returns
        -------
        pdraws: np.array
            index array of flattened samps chosed to sample at next level
        """
        dres = self.levels[self.idx]
        pres = self.scene.ptshape
        # direction detail
        daxes = tuple(range(len(pres), len(pres) + len(dres)))
        p = wavelet.get_detail(samps, daxes)
        p = p*(1 - self._sample_t) + np.median(p)*self._sample_t
        # draw on pdf
        nsampc = int(self._sample_rate*samps.size)
        pdraws = np.random.choice(p.size, nsampc, replace=False, p=p/np.sum(p))
        return pdraws

    def update_pdf(self, samps, si, lum):
        """update samps (which holds values used to calculate pdf)

        Parameters
        ----------
        si: np.array
            multidimensional indices to update
        lum:
            values to update with

        """
        self.skydetail = np.maximum(self.skydetail, np.max(lum, 0))
        samps[tuple(si)] = np.max(lum, 1)

    def save_pdf(self, samps):
        outf = f'{self.scene.outdir}/{self.stype}_pdf'
        np.save(outf, samps)
        outf = f'{self.scene.outdir}/{self.stype}_skydetail'
        np.save(outf, self.skydetail.reshape(self.skres, self.skres))
