# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import os
import pickle
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from raytraverse import io, optic


class LightField(object):
    """container for accessing sampled data

    Parameters
    ----------
    scene: raytraverse.scene.Scene
        scene class containing geometry, location and analysis plane
    rebuild: bool, optional
        build kd-tree even if one exists
    prefix: str, optional
        prefix of data files to map
    """

    def __init__(self, scene, rebuild=False, prefix='sky'):
        #: bool: force rebuild kd-tree
        self.rebuild = rebuild
        #: str: prefix of data files from sampler (stype)
        self.prefix = prefix
        self._vlo = None
        self._d_kd = None
        self.scene = scene

    @property
    def vlo(self):
        """direction vector (3,) luminance (srcn,), omega (1,)"""
        return self._vlo

    @property
    def d_kd(self):
        """list of direction kdtrees

        :getter: Returns kd tree structure
        :type: list of scipy.spatial.cKDTree
        """
        return self._d_kd

    @property
    def scene(self):
        """scene information

        :getter: Returns this integrator's scene
        :setter: Set this integrator's scene
        :type: raytraverse.scene.Scene
        """
        return self._scene

    @scene.setter
    def scene(self, scene):
        """Set this field's scene and load samples"""
        self._scene = scene
        kdfile = f'{scene.outdir}/{self.prefix}_kd_data.pickle'
        if os.path.isfile(kdfile) and not self.rebuild:
            f = open(kdfile, 'rb')
            self._d_kd = pickle.load(f)
            self._vlo = pickle.load(f)
            f.close()
        else:
            self._d_kd, self._vlo = self._mk_tree()
            f = open(kdfile, 'wb')
            pickle.dump(self.d_kd, f, protocol=4)
            pickle.dump(self.vlo, f, protocol=4)
            f.close()

    def _get_vl(self, npts, pref=''):
        dfile = f'{self.scene.outdir}/{self.prefix}{pref}_vals.out'
        vfile = f'{self.scene.outdir}/{self.prefix}{pref}_vecs.out'
        if not (os.path.isfile(dfile) and os.path.isfile(vfile)):
            raise FileNotFoundError("No results files found, have you run"
                                    f" a Sampler of type {self.prefix} for"
                                    f" scene {self.scene.outdir}?")
        fvecs = io.bytefile2np(open(vfile, 'rb'), (-1, 4))
        sorting = fvecs[:, 0].argsort()
        fvals = optic.rgb2rad(io.bytefile2np(open(dfile, 'rb'), (-1, 3)))
        fvals = fvals.reshape(fvecs.shape[0], -1)[sorting]
        fvecs = fvecs[sorting]
        pidx = fvecs[:, 0]
        pt_div = np.searchsorted(pidx, np.arange(npts), side='right')
        pt0 = 0
        vl = []
        for i, pt in enumerate(pt_div):
            vl.append(np.hstack((fvecs[pt0:pt, 1:4], fvals[pt0:pt])))
            pt0 = pt
        return vl

    def outfile(self, idx):
        istr = "_".join([f"{i:04d}" for i in np.asarray(idx).reshape(-1)])
        return f"{self.scene.outdir}_{self.prefix}_{istr}"

    def _mk_tree(self):
        return None, None

    def measure(self, pi, vecs, coefs=1, interp=1):
        """measure the source coefficients for a vector from point index
        and apply coefficients"""
        pass

    def items(self):
        return range(np.product(self.scene.ptshape))

    def _dview(self, idx, pdirs, mask, res=800):
        img = np.zeros((res, res*self.scene.view.aspect))
        lum = self.measure(idx, pdirs[mask])
        img[mask] = lum
        outf = f"{self.outfile(idx)}.hdr"
        io.array2hdr(img, outf)
        return outf

    def direct_view(self, res=800):
        """create a summary image of lightfield for each vpt"""
        vm = self.scene.view
        pdirs, mask = vm.pixelrays(res)
        fu = []
        with ThreadPoolExecutor() as exc:
            for idx in self.items():
                fu.append(exc.submit(self._dview, idx, pdirs, mask))
        [print(f.result()) for f in fu]
