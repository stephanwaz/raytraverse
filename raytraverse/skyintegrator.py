# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

import os
import shutil

import numpy as np
import clasp.script_tools as cst

from raytraverse import translate, Integrator


class SkyIntegrator(Integrator):
    """integrator for sky results has methods for generating sun sampling pdfs

    Parameters
    ----------
    scene: raytraverse.scene.Scene
        scene class containing geometry, location and analysis plane
    rebuild: bool, optional
        build kd-tree even if one exists
    """

    def __init__(self, scene, rebuild=False):
        super(SkyIntegrator, self).__init__(scene, rebuild, 'sky')

    def lift_samples(self, pidx, uv, lum, scheme, maxspec):
        lum = np.where(lum > maxspec, 0, lum)
        pdf = np.zeros(scheme[0, 0:4])
        l0 = 0
        pts = np.prod(self.scene.ptshape)
        for l in scheme:
            l1 = l0 + l[5]
            pdf = translate.resample(pdf, l[0:4]).reshape(pts, *l[2:4])
            ij = translate.uv2ij(uv[l0:l1], l[3]).reshape(-1, 2)
            si = np.vstack((pidx[l0:l1], ij.T)).astype(int)
            pdf[tuple(si)] = lum[l0:l1]
            pdf = pdf.reshape(l[0:4])
            l0 = l1
        return pdf

    def write_sun_pdfs(self, suns, maxspec=0.3, reload=True):
        """update sky_pdf to only consider sky patches with direct sun

        Parameters
        ----------
        suns
        maxspec: float, optional
        reload: bool, optional
        """
        outd = f'{self.scene.outdir}/sunpdfs'
        if not reload:
            shutil.rmtree(outd, ignore_errors=True)
        if not os.path.exists(outd):
            try:
                os.mkdir(outd)
            except FileExistsError:
                raise FileExistsError('sun pdfs already exists, use '
                                      'reload=False to regenerate')
            # recover sampling order
            isort = self.isort.argsort()
            scheme = np.load(f'{self.scene.outdir}/sky_scheme.npy').astype(int)
            side = np.sqrt(self.lum[0].shape[1])
            sunuv = translate.xyz2uv(suns)
            sunbin = translate.uv2bin(sunuv, side).astype(int)
            vec = np.array(self.vec).reshape(-1, 3)[isort]
            pidx = self.pidx[isort]
            uv = self.scene.view.xyz2uv(vec)
            lums = np.array(self.lum)
            lum = np.max(lums[:, :, sunbin], 2).flatten()[isort]
            pdf = self.lift_samples(pidx, uv, lum, scheme, maxspec)
            np.save(f'{self.scene.outdir}/sky_pdf', pdf)
            for i, sb in enumerate(sunbin):
                lum = lums[:, :, sb].flatten()[isort]
                pdf = self.lift_samples(pidx, uv, lum, scheme, maxspec)
                np.save(f'{outd}/{i:04d}_{sb:04d}', pdf)

    def write_skydetail(self, reload=False):
        outf = f'{self.scene.outdir}/sky_skydetail.npy'
        if os.path.isfile(outf) and reload:
            return np.load(outf)
        side = int(np.sqrt(self.lum[0].shape[1]))
        skydetail = np.max(np.array(self.lum), (0, 1)).reshape(side, side)
        np.save(outf, skydetail)
        return skydetail
