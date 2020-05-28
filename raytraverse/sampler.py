# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

import numpy as np


class Sampler(object):
    """execute sampling scheme according to serializers and engine

    to properly manage resources, Sampler should be used with the "with"
    statement::

        with Sampler('scene.rad', spos, sdir, ssrc) as sampler:
            while sampler.idx < sdir.shape - 1:
                result = sampler.run_samples()
                result2 = s.evaluate_samples(result)
                for nsamps, complete in result2:
                    ...
                sampler.idx += 1

    Parameters
    ----------
    scene: str
        space separated list of radiance scene files (no sky)
    spos: plenotraverse.serial.Serial.SerializePos
    sdir: plenotraverse.serial.Serial.SerializeDir
    ssrc: plenotraverse.serial.Serial.SerializeSrc
    pdraw: {'lower-left', 'random', 'centered', 'increment'}, optional
        drawtype for pos
    ddraw: {'lower-left', 'random', 'centered', 'increment'}, optional
        drawtype for dir
    contrast: float, optional
        ratio of max:average for testing var
    lmin: float, optional
        min value of max to resample
    nsamp: int, optional
        number of samples per sub bin (averaged by engine)
        no effect without jidx
    jidx: int, optional
        if given, the idx of serializer overwhich to jitter samples
    order: tuple
        must be ('dir', 'pos', 'src') in some order

    Attributes
    ----------
    scene
    spos: plenotraverse.serial.Serial.SerializePos
    sdir: plenotraverse.serial.Serial.SerializeDir
    ssrc: plenotraverse.serial.Serial.SerializeSrc
    pdraw: {'lower-left', 'random', 'centered', 'increment'}
        drawtype for pos
    ddraw: {'lower-left', 'random', 'centered', 'increment'}
        drawtype for dir
    contrast: float, optional
        ratio of max:average for testing var
    lmin: float, optional
        min value of max to resample
    nsamp: int
        number of samples per sub bin (averaged by engine)
        no effect without jidx
    jidx: int, optional
        if given, the idx of serializer overwhich to jitter samples
    idx
    """

    #: np.dtype: data type for storage  and sampling record arrays
    dt = np.dtype([('sb', '<u2'), ('du', '<f2'), ('dv', '<f2'),
                   ('pu', '<f2'), ('pv', '<f2'), ('l', '<f4'),
                   ('size1', 'u1'), ('size2', 'u1')])

    def __init__(self, scene, area, outdir, overwrite=False, skyres=16,
                 dsteps=(16, 32, 64, 128, 256, 512), psteps=(2, 4, 8, 16)):
        self.spos = spos
        self.sdir = sdir
        self.ssrc = ssrc
        self.order = order
        self.contrast = contrast
        self.lmin = lmin
        self.pdraw = pdraw
        self.ddraw = ddraw
        self._samples = None
        self._shape = None
        self.scene = scene
        self.nsamp = nsamp
        self.jidx = jidx
        self.idx = 0

    def __enter__(self):
        return self

    @property
    def scene(self):
        """render scene files

        :getter: Returns this samplers's scene file path (deleted on exit)
        :setter: Sets this samplers's scene file path and creates necesarry run files
        :type: str
        """
        return self._scene

    @scene.setter
    def scene(self, scene):
        self._scene = scene

    @property
    def idx(self):
        """controls which level of serializers to sample

        :getter: Returns this samplers's current serializer.shape index
        :setter: Sets this samplers's index and prepares sampler for running
            and evaluation at this level. Note that idx cannot be the last
            position of spos and sdir
        :type: int

        """
        return self._idx

    @idx.setter
    def idx(self, idx):
        dbins = self.sdir.binlist(idx + 1, draw=self.ddraw)
        pbins, pshape = self.spos.binlist(idx + 1, draw=self.pdraw, testi=-1)
        duv = self.sdir.bin2uv(dbins)
        puv = self.spos.bin2uv(pbins)
        self._samples = self.meshgrid_md(duv, puv, shape=(-1, 4))
        self._idx = idx
        self._shape = np.square(self.sdir.shape[idx:idx + 2] +
                                self.spos.shape[idx:idx + 2] +
                                self.ssrc.shape[idx:idx + 2])
        # to account for double square repr of dir
        # src is clipped to upper hemisphere, so has same number as square
        self._shape[0] *= 2
        # to account for point area filtering
        self._shape[2] = pshape
        self._e_shape = self._shape[self._e_order, ]
        self._e_box = np.append(np.cumprod(self._e_shape[-1:0:-1])[::-1], 1)
        srz = {'dir': self.sdir.boxbin, 'pos': self.spos.boxbin,
               'src': self.ssrc.boxbin}
        self._box_bins = tuple([srz[i][self.idx] for i in self.order] +
                               [srz[i][self.idx + 1] for i in self.order])

    def run_samples(self):
        """virtual function
        """
        return None, None

    def _dump(self, recar, lum, samp, size, j):
        """add luminance samples to recarray

        Parameters
        ----------
        recar: np.recarray
            array to write to
        lum: np.array
            luminance values to write
        samp: np.array
            samples matching luminance values
        size:
            index of shape samples correspond to
        j: int
            counter to increment with samples stored

        Returns
        -------
        int
            updated counter

        """
        samp = samp.reshape(-1, 5)
        lum = lum.flatten()
        smp = lum.size
        ss = self._size_i('src', size)
        ds = self._size_i('dir', size)
        ps = self._size_i('pos', size)
        bsize = io.omegas2byte(ss, ds, ps)
        recar['size1'][j:j + smp] = bsize[:, 0]
        recar['size2'][j:j + smp] = bsize[:, 1]
        recar['l'][j:j + smp] = lum
        recar['du'][j:j + smp] = samp[:, 0]
        recar['dv'][j:j + smp] = samp[:, 1]
        recar['pu'][j:j + smp] = samp[:, 2]
        recar['pv'][j:j + smp] = samp[:, 3]
        recar['sb'][j:j + smp] = samp[:, 4]
        return j + smp

    def __exit__(self, exc_type, exc_value, traceback):
        """delete temporary octree file"""
        os.remove(self._scene)
