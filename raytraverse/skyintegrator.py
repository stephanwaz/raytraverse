# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

import os
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from raytraverse import translate, Integrator, io


class SkyIntegrator(Integrator):
    """loads scene and sampling data for processing

    Parameters
    ----------
    scene: raytraverse.scene.Scene
        scene class containing geometry, location and analysis plane
    rebuild: bool, optional
        build kd-tree even if one exists
    prefix: str, optional
        prefix of data files to integrate
    """

    def __init__(self, scene, rebuild=False):
        super(SkyIntegrator, self).__init__(scene, rebuild, 'sky')

    def filter_sky_pdf(self, suns, maxspec=0.3, reload=False):
        """update sky_pdf to only consider sky patches with direct sun

        Parameters
        ----------
        suns
        maxspec: float, optional
        """
        outf = f'{self.scene.outdir}/sky_pdf.npy'
        if os.path.isfile(outf) and reload:
            return np.load(outf)
        isort = self.isort.argsort()
        scheme = np.load(f'{self.scene.outdir}/sky_scheme.npy').astype(int)
        side = np.sqrt(self.lum[0].shape[1])
        sunuv = translate.xyz2uv(suns)
        sunbin = translate.uv2bin(sunuv, side)
        sunbin = np.unique(sunbin).astype(int)
        lum = np.max(np.array(self.lum)[:, :, sunbin], 2).flatten()[isort]
        lum = np.where(lum > maxspec, 0, lum)
        vec = np.array(self.vec).reshape(-1, 3)[isort]
        pidx = self.pidx[isort]
        uv = self.scene.view.xyz2uv(vec)
        pts = np.prod(self.scene.ptshape)
        pdf = np.zeros(scheme[0, 0:4])
        l0 = 0
        for l in scheme:
            l1 = l0 + l[4]
            pdf = translate.resample(pdf, l[0:4]).reshape(pts, *l[2:4])
            ij = translate.uv2ij(uv[l0:l1], l[3]).reshape(-1, 2)
            si = np.vstack((pidx[l0:l1], ij.T)).astype(int)
            pdf[tuple(si)] = lum[l0:l1]
            pdf = pdf.reshape(l[0:4])
            l0 = l1
        np.save(outf, pdf)
        return pdf
