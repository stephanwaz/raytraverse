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
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import matplotlib.pyplot as plt

from clasp import script_tools as cst
from raytraverse import translate, optic, io, wavelet


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


class Sampler(object):
    """holds scene information and sampling scheme

    Parameters
    ----------
    scene: raytraverse.scene.Scene
        scene class containing geometry, location and analysis plane
    ptres: float, optional
        final spatial resolution in scene geometry units
    dndepth: int, optional
        final directional resolution given as log2(res)
    skres: int, optional
        side of square sky resolution (must be even)
    sunsperpatch: int, optional
        maximum number of suns per sky patch to sample
    t0: float, optional
        in range 0-1, fraction of uniform random samples taken at first step
    t1: float, optional
        in range 0-t0, fraction of uniform random samples taken at final step
    minrate: float, optional
        in range 0-1, fraction of samples at final step (this is not the total
        sampling rate, which depends on the number of levels).
    ipres: int, optional
        minimum position resolution (across maximum length of area)
    idres: int, optional
        initial direction resolution (as log2(res))
    """

    def __init__(self, scene, ptres=1.0, dndepth=9, skres=20, sunsperpatch=4,
                 t0=.1, t1=.01, minrate=.05, idres=4, ipres=4):
        self.scene = scene
        #: int: minimum position resolution (across maximum length of area)
        self.ipres = ipres
        #: int: initial direction resolution (as log2(res))
        self.idres = idres
        self.levels = (ptres, dndepth, skres)
        #: np.array: shape of current sampling level
        self.current_level = self.levels[0]
        #: int: maximum number of suns per skypatch
        self.sunsperpatch = sunsperpatch
        #: float: fraction of uniform random samples taken at first step
        self.t0 = t0
        #: float: fraction of uniform random samples taken at final step
        self.t1 = t1
        #: float: fraction of samples at final step
        self.minrate = minrate
        #: bool: set to True after call to this.mkpmap
        self.skypmap = os.path.isfile(f"{scene.outdir}/sky.gpm")
        #: int: index of next level to sample
        self.idx = 0

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
        self.current_level = self.levels[idx]
        x = self.idx/(self.levels.shape[1]-1)
        self._sample_t = wavelet.get_uniform_rate(x, self.t0, self.t1)
        self._sample_rate = wavelet.get_sample_rate(x, self.minrate)

    @property
    def levels(self):
        """sampling scheme

        :getter: Returns the sampling scheme
        :setter: Set the sampling scheme from (ptres, dndepth, skres)
        :type: np.array
        """
        return self._levels

    @levels.setter
    def levels(self, res):
        """calculate sampling scheme"""
        ptres, dndepth, skres = res
        bbox = self.scene.area.bbox[:, 0:2]/ptres
        size = (bbox[1] - bbox[0])
        uvlevels = np.floor(np.log2(np.max(size)/self.ipres)).astype(int)
        uvpow = 2**uvlevels
        uvsize = np.ceil(size/uvpow)*uvpow
        plevels = np.stack([uvsize/2**(uvlevels-i)
                            for i in range(uvlevels+1)])
        dlevels = np.array([(2**(i+1), 2**i)
                            for i in range(self.idres, dndepth+1, 1)])
        plevels = np.pad(plevels, [(0, dlevels.shape[0] - plevels.shape[0]),
                                   (0, 0)], mode='edge')
        slevels = np.full(dlevels.shape, skres)
        self._levels = np.hstack((plevels, dlevels, slevels)).astype(int)

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
        skyoct = f'{self.scene.outdir}/sky.oct'
        if not os.path.isfile(skyoct):
            skydef = ("void light skyglow 0 0 3 1 1 1 skyglow source sky 0 0 4"
                      " 0 0 1 180")
            f = open(f'{self.scene.outdir}/sky.oct', 'wb')
            cst.pipeline([f'oconv -i {self.scene.outdir}/scene.oct -'], inp=skydef,
                          outfile=f, close=True)
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
               f'{fdr}/sky.gpm {nphotons} {fdr}/sky.oct')
        r, err = cst.pipeline([cmd], caperr=True)
        if b'fatal' in err:
            raise ChildProcessError(err.decode(cst.encoding))
        self.skypmap = True
        return cst.pipeline([f'getinfo {fdr}/sky.gpm'])

    def sky_sample(self, vecs, rcopts='-ab 7 -ad 60000 -as 30000 -lw 1e-7',
                   nproc=12, executable='rcontrib_pm', bwidth=1000):
        fdr = self.scene.outdir
        if self.skypmap:
            rcopts += f' -ab -1 -ap {fdr}/sky.gpm {bwidth}'
        side = self.levels[self.idx, -1]
        rc = (f"{executable} -V+ -fff {rcopts} -h -n {nproc} -e "
              f"'side:{side}' -f {fdr}/scbins.cal -b bin -bn {side**2} "
              f"-m skyglow {fdr}/sky.oct")
        p = Popen(shlex.split(rc), stdout=PIPE,
                  stdin=PIPE).communicate(io.np2bytes(vecs))
        lum = optic.rgb2lum(io.bytes2np(p[0], (-1, 3)))
        return lum.reshape(-1, side**2)

    def sample_idx(self, pdraws, upsample=True):
        shape = self.current_level[0:4]
        # index assignment
        # scale and repeat if upsampling
        if upsample:
            oshape = self.levels[self.idx - 1, 0:4]
            si = np.stack(np.unravel_index(pdraws, oshape))
            rs = (shape/oshape).astype(int)
            si *= rs[:, None]
            # repeat along position and direction axes
            for i in range(4):
                si = np.repeat(si[:, None, :], rs[i], 1)
                si[i] += np.arange(rs[i]).astype(np.int64)[:, None]
                si = si.reshape(4, -1)
        else:
            si = np.stack(np.unravel_index(pdraws, shape))
        # convert to UV directions and positions
        uv = si.T[:, 2:]/shape[3]
        pos = self.scene.area.uv2pt(si.T[:, 0:2])
        uv += np.random.random(uv.shape)/shape[2]
        xyz = translate.uv2xyz(uv, axes=(0, 2, 1))
        vecs = np.hstack((pos, xyz))
        return si, vecs

    def dump(self, vecs, vals, wait=False):
        prefix = f'{self.scene.outdir}/sky'
        if wait:
            io.write_npy(vecs, vals, self.current_level, prefix)
        else:
            executor = ThreadPoolExecutor()
            executor.submit(io.write_npy, vecs, vals, self.current_level,
                            prefix)

    def draw(self, samps):
        # detail is calculated across position and direction seperately and
        # combined by product (would be summed otherwise) to avoid drowning out
        # the signal in the more precise dimensions (assuming a mismatch in step
        # size and final stopping criteria
        p = np.ones(samps.size)
        self.idx += 1
        dres = self.current_level[2:4]
        pres = self.current_level[0:2]
        # direction detail
        if dres[0] > samps.shape[-1]:
            daxes = tuple(range(len(pres), len(pres) + len(dres)))
            p = p*wavelet.get_detail(samps, daxes)
        # position detail
        if pres[0] > samps.shape[0]:
            paxes = tuple(range(len(pres)))
            p = p*wavelet.get_detail(samps, paxes)

        p = p*(1 - self._sample_t) + np.median(p)*self._sample_t
        # draw on pdf
        nsampc = int(self._sample_rate*samps.size)
        pdraws = np.random.choice(p.size, nsampc, replace=False, p=p/np.sum(p))
        return pdraws

    def run(self, **skwargs):
        allc = 0
        samps = np.zeros(self.current_level[0:4])
        for i in range(self.idx, self.levels.shape[0]):
            skbins = self.current_level[4]**2
            if i == 0:
                draws = np.arange(np.prod(self.current_level[0:4]))
                upsample = False
            else:
                draws = self.draw(samps)
                samps = translate.resample(samps, self.current_level[0:4])
                upsample = True
            si, vecs = self.sample_idx(draws, upsample=upsample)
            srate = si.shape[1]/np.prod(self.current_level[0:4])
            print(f"{self.current_level} sampling: {si.shape[1]}\t{srate:.02%}")
            lum = self.sky_sample(vecs, **skwargs)
            self.dump(vecs, lum)
            samps[tuple(si)] = np.sum(lum, 1)
            print(samps.shape)
            a = lum.size
            fig, axes = plt.subplots(1, 1, figsize=[20, 10])
            io.imshow(axes, np.log10(samps[-1, -1]/179), cmap=plt.cm.viridis,
                      vmin=-3,
                      vmax=0)
            plt.tight_layout()
            plt.show()
            allc += a
        print("--------------------------------------")
        srate = allc/(samps.size*skbins)
        print(f"asamp: {allc}\t{srate:.02%}")


