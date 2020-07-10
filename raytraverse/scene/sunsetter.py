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

from raytraverse import wavelet, translate, io, optic, plot
from raytraverse.mapper import SunMapper


class SunSetter(object):
    """select suns to sample based on sky pdf and scene.

    Parameters
    ----------
    scene: raytraverse.scene.Scene
        scene class containing geometry, location and analysis plane
    srct: float, optional
        threshold of sky contribution for determining appropriate srcn
    """

    def __init__(self, scene, srct=.01, reload=True, **kwargs):
        #: float: threshold of sky contribution for determining appropriate srcn
        self.srct = srct
        #: bool: reuse existing sun positions (if found)
        self.reload = reload
        #: raytraverse.scene.Scene
        self.scene = scene
        sunfile = f"{self.scene.outdir}/suns.rad"
        self._suns = self.init_suns(sunfile)
        self.map = SunMapper(self.suns)
        if not (os.path.isfile(sunfile) and self.reload):
            self._write_suns(sunfile)

    @property
    def suns(self):
        """holds sun positions

        :getter: Returns the sun source array
        :setter: Set the sun source array and write to files
        :type: np.array
        """
        return self._suns

    def init_suns(self, sunfile):
        """set the skydetail array and determine sample count and spacing"""
        if os.path.isfile(sunfile) and self.reload:
            f = open(sunfile, 'r')
            sund = f.read().split('source')[1:]
            xyz = np.array([s.split()[4:7] for s in sund]).astype(float)
        else:
            uvsize = self.scene.skyres
            skyb = self.load_sky_facs()
            si = np.stack(np.unravel_index(np.arange(skyb.size), skyb.shape))
            uv = si.T/uvsize
            ib = self.scene.in_solarbounds(uv + .5/uvsize,
                                           size=1/uvsize).reshape(skyb.shape)
            suncount = np.sum(skyb*ib > self.srct)
            skyd = wavelet.get_detail(skyb, (0, 1)).reshape(skyb.shape)
            sb = (skyb + skyd)/np.min(skyb + skyd)
            sd = (sb * ib).ravel()
            sdraws = np.random.default_rng().choice(skyb.size, suncount,
                                                    replace=False,
                                                    p=sd/np.sum(sd))
            si = np.stack(np.unravel_index(sdraws, skyb.shape))
            # keep solar discs from overlapping
            border = .3*uvsize/180
            j = np.random.default_rng().uniform(border, 1-border, si.T.shape)
            uv = (si.T + j)/uvsize
            xyz = translate.uv2xyz(uv, xsign=1)
        return xyz

    def load_sky_facs(self):
        outf = f'{self.scene.outdir}/sky_skydetail.npy'
        if os.path.isfile(outf) and self.reload:
            return np.load(outf)
        dfile = f'{self.scene.outdir}/sky_vals.out'
        fvals = optic.rgb2rad(io.bytefile2np(open(dfile, 'rb'), (-1, 3)))
        sd = np.max(fvals.reshape(-1, self.scene.skyres, self.scene.skyres), 0)
        np.save(outf, sd)
        return sd

    def write_sun_pdfs(self, reload=True):
        """update sky_pdf to only consider sky patches with direct sun

        Parameters
        ----------
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
            scheme = np.load(f'{self.scene.outdir}/sky_scheme.npy').astype(int)
            side = self.scene.skyres
            sunuv = translate.xyz2uv(self.suns)
            sunbin = translate.uv2bin(sunuv, side).astype(int)
            dfile = f'{self.scene.outdir}/sky_vals.out'
            fvals = optic.rgb2rad(io.bytefile2np(open(dfile, 'rb'), (-1, 3)))
            vfile = f'{self.scene.outdir}/sky_vecs.out'
            fvecs = io.bytefile2np(open(vfile, 'rb'), (-1, 4))
            vec = fvecs[:, 1:4]
            pidx = fvecs[:, 0]
            uv = self.scene.view.xyz2uv(vec)
            lums = fvals.reshape(-1, side*side)
            lum = np.max(lums[:, sunbin], 1)
            pdf = self._lift_samples(pidx, uv, lum, scheme, self.scene.maxspec)
            np.save(f'{self.scene.outdir}/sky_pdf', pdf)
            for i, sb in enumerate(sunbin):
                lum = lums[:, sb]
                pdf = self._lift_samples(pidx, uv, lum, scheme,
                                         self.scene.maxspec)
                np.save(f'{outd}/{i:04d}_{sb:04d}', pdf)

    def direct_view(self):
        sxy = translate.xyz2xy(self.suns, flip=False)
        lums, fig, ax, norm, lev = plot.mk_img_setup([0, 1], ext=1)
        ax.scatter(sxy[:, 0], sxy[:, 1], s=15,
                   marker='o', facecolors='yellow')
        ax.set_facecolor((0, 0, 0))
        outf = f"{self.scene.outdir}_suns.png"
        plot.save_img(fig, ax, outf)

    def write_sun(self, i):
        s = self.suns[i]
        mod = f"solar{i:05d}"
        name = f"sun{i:05d}"
        d = f"{s[0]} {s[1]} {s[2]}"
        dec = f"void light {mod} 0 0 3 1 1 1\n"
        dec += f"{mod} source {name} 0 0 4 {d} 0.533\n"
        return dec, mod

    def _write_suns(self, sunfile):
        """write suns to file

        Parameters
        ----------
        sunfile
        """
        f = open(sunfile, 'w')
        g = open(f'{self.scene.outdir}/sun_modlist.txt', 'w')
        for i in range(self.suns.shape[0]):
            dec, mod = self.write_sun(i)
            print(dec, file=f)
            print(mod, file=g)
        f.close()
        g.close()

    def _lift_samples(self, pidx, uv, lum, scheme, maxspec):
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
