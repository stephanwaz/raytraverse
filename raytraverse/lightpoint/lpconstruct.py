# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
"""functions for generating new lightpoints from existing"""
import numpy as np

from raytraverse.lightpoint.lightpointkd import LightPointKD


def add_sources(lf1, lf2, src=None, calcomega=True, write=True):
    """add light points of distinct sources together
    results in a new lightpoint with srcn=srcn1+srcn2 and
    vector size=vecsize1+vecsize2

    Parameters
    ----------
    lf1: raytraverse.lightpoint.LightPointKD
        this lightpoint sets all parameters of output
    lf2: raytraverse.lightpoint.LightPointKD
    src: str, optional
        if None (default), src is "{lf1.src}_{lf2.src}"
    calcomega: bool, optional
        passed to LightPointKD constructor
    write: bool, optional
        passed to LightPointKD constructor
    Returns
    -------
    raytraverse.lightpoint.LightPointKD
    """
    vecs = np.concatenate((lf1.vec, lf2.vec), axis=0)
    i, d = lf1.query_ray(lf2.vec)
    lum1at2 = np.concatenate((lf1.lum, lf1.lum[i]), axis=0)
    i, d = lf2.query_ray(lf1.vec)
    lum2at1 = np.concatenate((lf2.lum[i], lf2.lum), axis=0)
    lums = np.concatenate((lum1at2, lum2at1), axis=1)
    if src is None:
        src = f"{lf1.src}_{lf2.src}"
    lf_out = LightPointKD(lf1.scene, vec=vecs, lum=lums, vm=lf1.vm, pt=lf1.pt,
                          posidx=lf1.posidx, src=src, srcn=lf1.srcn + lf2.srcn,
                          calcomega=calcomega, write=write)
    return lf_out
