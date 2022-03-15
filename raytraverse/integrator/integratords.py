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
from raytraverse.lightpoint import LightPointKD


def _prep_ds(lpts, skyvecs):
    try:
        snp = lpts[2]
    except IndexError:
        # this implies the sunpt is not needed (0 direct solar)
        return lpts[0:1], skyvecs[0:1], None
    skyvecs = [np.hstack(skyvecs[0:2]), skyvecs[2]]
    ski, skyvecs, refl = intg.apply_dsky_patch(lpts[0], lpts[1], skyvecs,
                                               snp.srcdir)
    lpts = [ski, snp]
    return lpts, skyvecs, refl


def evaluate_pt_ds(lpts, skyvecs, suns, **kwargs):
    lpts, skyvecs, refl = _prep_ds(lpts, skyvecs)
    return intg.evaluate_pt(lpts, skyvecs, suns, refl=refl, **kwargs)


def img_pt_ds(lpts, skyvecs, suns, **kwargs):
    lpts, skyvecs, refl = _prep_ds(lpts, skyvecs)
    return intg.img_pt(lpts, skyvecs, suns, refl=refl, **kwargs)


class IntegratorDS(Integrator):
    """specialized integrator for 2-phase DDS style calculation. assumes
    first lightplane is sky contrribution, second, direct sky contribution
    (with identical sampling to sky) and third direct sun contribution. Uses
    special point functions that combine two sky functions on a per patch basis.

    Parameters
    ----------
    skplane: raytraverse.lightfield.LightPlaneKD
    snplane: raytraverse.lightfiled.SunsPlaneKD
    dskplane: raytraverse.lightfield.LightPlaneKD
    """
    evaluate_pt = evaluate_pt_ds
    img_pt = img_pt_ds

    def __init__(self, skplane, dskplane, snplane, sunviewengine=None):
        super(IntegratorDS, self).__init__(skplane, dskplane, snplane,
                                           sunviewengine=sunviewengine)

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
        skydatas = [smtx, dsns[:, 4:], dsns[:, 3:4]]
        return skydatas, dsns

