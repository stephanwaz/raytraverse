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
from raytraverse import optic, io, wavelet, Sampler
from raytraverse.sampler import scbinscal


class SCBinSampler(Sampler):
    """sample contributions from the sky hemisphere according to a square grid
    transformed by shirley-chiu mapping using rcontrib.

    Parameters
    ----------
    scene: raytraverse.scene.Scene
        scene class containing geometry, location and analysis plane
    srcn: int, optional
        side of square sky resolution
    """

    def __init__(self, scene, srcn=20, **kwargs):
        #: int: side of square sky resolution
        self.skres = srcn
        super(SCBinSampler, self).__init__(scene, srcn=srcn**2, stype='sky',
                                           **kwargs)
        #: bool: set to True after call to this.mkpmap
        self.skypmap = os.path.isfile(f"{scene.outdir}/sky.gpm")
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
