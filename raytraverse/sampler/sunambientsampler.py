# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import os

import numpy as np

import clasp.script_tools as cst

from raytraverse.sampler.sampler import Sampler


class SunAmbientSampler(Sampler):
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
        super().__init__(scene, stype='sunambient', fdres=fdres, t0=0, t1=0,
                         minrate=minrate, maxrate=maxrate, idres=idres,
                         **kwargs)
        # need to check for octree because mkpmap checks data rather than size
        # for octree staleness
        if not os.path.isfile(self.compiledscene):
            self.compiledscene = f"{scene.outdir}/suns.rad"
        self.srcn = self.suns.suns.shape[0]
        self.skypmap = os.path.isfile(f'{scene.outdir}/{self.stype}.gpm')

    def __del__(self):
        pass

    def sample(self, vecs, rcopts='-ab 7 -ad 60000 -as 30000 -lw 1e-7',
               nproc=12, executable='rcontrib', pmexecutable='rcontrib_pm',
               usepmap=True, bps=200):
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
        pmexecutable: str, optional
            rendering engine binary for photon map use
        usepmap: bool, optional
            set to false to override use of any existing sun.gpm
        bps: int, optional
            if using photon mapping, the bandwidth per source

        Returns
        -------
        lum: np.array
            array of shape (N,) to update weights
        """
        if self.skypmap and usepmap:
            bwidth = int(bps*self.srcn)
            rcopts += f' -ab -1 -ap {self.scene.outdir}/{self.stype}.gpm {bwidth}'
            executable = pmexecutable
        rc = (f"{executable} -fff {rcopts} -h -n {nproc} "
              f"-M {self.scene.outdir}/sun_modlist.txt {self.compiledscene}")
        lum = super().sample(vecs, call=rc)
        return np.max(lum.reshape(-1, self.srcn), 1)

    def mkpmap(self, apo=[], nproc=12, overwrite=False, fps=2e6,
               executable='mkpmap_dc', opts=''):
        """makes photon map of skydome with specified photon port modifier

        Parameters
        ----------
        apo: list, optional
            list of photon port modifiers, if not given or empty runs without
            ports
        nproc: int, optional
            number of processes to run on (the -n option of mkpmap)
        overwrite: bool, optional
            if True, passes -fo+ to mkpmap, if false and pmap exists, raises
            ChildProcessError
        fps: int, optional
            number of contribution photons per sun
        executable: str, optional
            path to mkpmap executable
        opts: str, optional
            additional options to feed to mkpmap

        Returns
        -------
        str
            result of getinfo on newly created photon map
        """
        nphotons = int(fps*self.srcn)
        if len(apo) > 0:
            apos = '-apo ' + ' -apo '.join(apo)
        else:
            apos = ''
        if overwrite:
            force = '-fo+'
        else:
            force = '-fo-'
        fdr = self.scene.outdir
        cmd = (f'{executable} {opts} -n {nproc} {force} {apos} -apC '
               f'{fdr}/{self.stype}.gpm {nphotons} {fdr}/{self.stype}.oct')
        r, err = cst.pipeline([cmd], caperr=True)
        print(cmd)
        if b'fatal' in err:
            print(err.decode(cst.encoding))
        self.skypmap = True
        return cst.pipeline([f'getinfo {fdr}/{self.stype}.gpm'])
