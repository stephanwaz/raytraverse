# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import numpy as np

from raytraverse import translate
from raytraverse.integrator.integrator import Integrator
import raytraverse.integrator._helpers as intg


def _prep_dv(lpts, skyvecs, skdir):
    dirlum = np.zeros((len(lpts[0].lum), 1))
    ski, skyvecs, refl = intg.apply_dsky_patch(lpts[0], lpts[1], skyvecs,
                                               skdir, dirlum)
    lpts = [ski]
    return lpts, skyvecs, refl


def evaluate_pt_dv(lpts, skyvecs, suns, **kwargs):
    side = int((skyvecs[0].shape[1] - 1)**.5)
    skpatch = translate.xyz2skybin(suns[0], side)[0]
    skdir = translate.skybin2xyz(skpatch, side)[0]
    lpts, skyvecs, refl = _prep_dv(lpts, skyvecs, skdir)
    return intg.evaluate_pt(lpts, skyvecs, suns, refl=refl, **kwargs)


def img_pt_dv(lpts, skyvecs, suns, **kwargs):
    side = int((skyvecs[0].shape[1] - 1)**.5)
    skpatch = translate.xyz2skybin(suns[0], side)[0]
    skdir = translate.skybin2xyz(skpatch, side)[0]
    lpts, skyvecs, refl = _prep_dv(lpts, skyvecs, skdir)
    return intg.img_pt(lpts, skyvecs, suns, refl=refl, **kwargs)


class IntegratorDV(Integrator):
    """specialized integrator for 2-phase Direct Views style calculation. assumes
    first lightplane is sky contrribution, second, direct sky contribution. Uses
    special point functions that combine two sky functions on a per patch basis.

    Parameters
    ----------
    skplane: raytraverse.lightfield.LightPlaneKD
    dskplane: raytraverse.lightfield.LightPlaneKD
    """
    evaluate_pt = evaluate_pt_dv
    img_pt = img_pt_dv

    def __init__(self, skplane, dskplane, sunviewengine):
        super(IntegratorDV, self).__init__(skplane, dskplane,
                                           sunviewengine=sunviewengine)

    def _group_query(self, skydata, points):
        # query and group sun positions
        gshape = (len(skydata.maskindices), len(points))
        suns = skydata.sun[:, None, 0:3]
        pts = np.atleast_2d(points)[None, :]
        vecs = np.concatenate(np.broadcast_arrays(suns, pts), -1).reshape(-1, 6)
        skydatas, dsns = self._unroll_sky_grid(skydata, gshape)
        idxs = []
        for lp in self.lightplanes:
            if lp.vecs.shape[1] == 6:
                idxs.append(lp.query(vecs)[0])
            else:
                idx, err = lp.query(points)
                idxs.append(np.broadcast_to(idx, gshape).ravel())
        idxs.append(np.broadcast_to(skydata.sunproxy[:, None], gshape).ravel())
        return np.stack(idxs), skydatas, dsns, vecs

    def _unroll_sky_grid(self, skydata, oshape):
        """

        Parameters
        ----------
        skydata: raytraverse.sky.SkyData
        oshape

        Returns
        -------

        """
        # broadcast skydata to match full indexing
        s = (*oshape[0:2], skydata.smtx.shape[1])
        smtx = np.broadcast_to(skydata.smtx[:, None, :], s).reshape(-1, s[-1])
        ds = (*oshape[0:2], skydata.sun.shape[1])
        dsns = np.broadcast_to(skydata.sun[:, None, :], ds).reshape(-1, ds[-1])
        # smtx (with sun), (patch sun, sun)
        skydatas = [np.hstack((smtx, dsns[:, 4:])), dsns[:, 3:4]]
        return skydatas, dsns
