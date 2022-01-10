# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import numpy as np

from raytraverse.integrator.integrator import Integrator


class IntegratorDS(Integrator):
    """collection of lightplanes with KDtree structure for sun position query

    Parameters
    ----------
    skplane: raytraverse.lightfield.LightPlaneKD
    snplane: raytraverse.lightfiled.SunsPlaneKD
    dskplane: raytraverse.lightfield.LightPlaneKD
    """

    def __init__(self, skplane, dskplane, snplane):
        super().__init__(skplane, dskplane, snplane)

    def _build_run_data(self, idxs, skydata, oshape):
        # broadcast skydata to match full indexing
        s = (*oshape[0:2], skydata.smtx.shape[1])
        smtx = skydata.smtx_patch_sun(includesky=True)
        dmtx = -skydata.smtx_patch_sun(includesky=False)
        smtx = np.broadcast_to(smtx[:, None, :], s).reshape(-1, s[-1])
        dmtx = np.broadcast_to(dmtx[:, None, :], s).reshape(-1, s[-1])
        ds = (*oshape[0:2], skydata.sun.shape[1])
        dsns = np.broadcast_to(skydata.sun[:, None, :], ds).reshape(-1, ds[-1])
        # makes coefficient list and fill idx lists
        tidxs = [np.broadcast_to(idxs[0], oshape[0:2]).ravel(),
                 np.broadcast_to(idxs[1], oshape[0:2]).ravel(), idxs[2]]
        skydatas = [smtx, dmtx, dsns[:, 3:4]]
        return tidxs, skydatas, dsns

