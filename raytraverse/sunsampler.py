# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import numpy as np

import clasp.script_tools as cst

from raytraverse import io, Sampler, translate

from memory_profiler import profile


class SunSampler(Sampler):
    """sample contributions from direct suns.

    Parameters
    ----------
    scene: raytraverse.scene.Scene
        scene class containing geometry, location and analysis plane
    suns: raytraverse.sunsetter.SunSetter
        sun class containing sun locations.
    maxspec: float
        maximum expected value of non reflected and/or scattered light
    """

    def __init__(self, scene, suns, idres=10,
                 fdres=12, maxrate=.01, minrate=.005,
                 **kwargs):
        self.suns = suns
        super(SunSampler, self).__init__(scene, stype='sunreflect',
                                         fdres=fdres,
                                         t0=0, t1=0, minrate=minrate,
                                         maxrate=maxrate, idres=idres, **kwargs)
        self.srcn = self.suns.suns.shape[0]
        self.init_weights()

    def init_weights(self):
        shape = np.concatenate((self.scene.ptshape, self.levels[self.idx]))
        try:
            skypdf = np.load(f"{self.scene.outdir}/sky_pdf.npy")
        except FileNotFoundError:
            print('Warning! sunsampler initialized without vector weights')
        else:
            self.weights = np.clip(translate.resample(skypdf, shape), 0, 1)



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
            array of shape (N,) to update weights
        """
        fdr = self.scene.outdir
        octr = f"{fdr}/sun.oct"
        rc = (f"{executable} -fff {rcopts} -h -n {nproc} "
              f"-M {fdr}/sun_modlist.txt {octr}")
        outf = f'{self.scene.outdir}/{self.stype}_vals.out'
        lum = io.call_sampler(outf, rc, vecs)
        return np.max(lum.reshape(-1, self.srcn), 1)

    def mkpmap(self, apo, nproc=12, overwrite=False, nphotons=1e8,
               executable='mkpmap_dc', opts=''):
        """makes photon map of skydome with specified photon port modifier

        Parameters
        ----------
        apo: str
            space seperated list of photon port modifiers
        nproc: int, optional
            number of processes to run on (the -n option of mkpmap)
        overwrite: bool, optional
            if True, passes -fo+ to mkpmap, if false and pmap exists, raises
            ChildProcessError
        nphotons: int, optional
            number of contribution photons
        executable: str, optional
            path to mkpmap executable
        opts: str, optional
            additional options to feed to mkpmap

        Returns
        -------
        str
            result of getinfo on newly created photon map
        """
        apos = '-apo ' + ' -apo '.join(apo.split())
        if overwrite:
            force = '-fo+'
        else:
            force = '-fo-'
        fdr = self.scene.outdir
        cmd = (f'{executable} {opts} -n {nproc} {force} {apos} -apC '
               f'{fdr}/sky.gpm {nphotons} {fdr}/sky_pm.oct')
        r, err = cst.pipeline([cmd], caperr=True)
        if b'fatal' in err:
            raise ChildProcessError(err.decode(cst.encoding))
        self.skypmap = True
        return cst.pipeline([f'getinfo {fdr}/sky.gpm'])

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
        p = self.weights.ravel()
        nsampc = int(self._sample_rate*self.weights.size)
        # draw on pdf
        pdraws = np.random.default_rng().choice(p.size, nsampc, replace=False,
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
        self.weights[tuple(si)] += lum
