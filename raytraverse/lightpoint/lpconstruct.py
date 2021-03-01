# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
"""functions for generating new lightpoints from existing"""
import numpy as np
from sklearn.cluster import DBSCAN

from raytraverse import translate
from raytraverse.lightpoint.lightpointkd import LightPointKD
from raytraverse.lightpoint.sunpointkd import SunPointKD


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
    kwargs = dict(vec=vecs, lum=lums, vm=lf1.vm, pt=lf1.pt, posidx=lf1.posidx,
                  src=src, srcn=lf1.srcn + lf2.srcn, calcomega=calcomega,
                  write=write)
    if hasattr(lf1, "sunview") and hasattr(lf1, "sunpos"):
        sunview = lf1.sunview
        sun = lf1.sunpos
    elif hasattr(lf2, "sunview") and hasattr(lf2, "sunpos"):
        sunview = lf2.sunview
        sun = lf2.sunpos
    else:
        sunview = None
        sun = None
    if sunview is None:
        lf_out = LightPointKD(lf1.scene, **kwargs)
    else:
        lf_out = SunPointKD(lf1.scene, sun=sun, sunview=sunview,
                            filterview=False, **kwargs)
    return lf_out


def _cluster(x, eps, min_samples=10):
    clust = DBSCAN(eps=eps, min_samples=min_samples)
    clust.fit(x)
    lsort = np.argsort(clust.labels_)
    ul, sidx = np.unique(clust.labels_[lsort], return_index=True)
    return np.array_split(lsort, sidx[1:])


def consolidate(lf, src=None, write=True, unit_eps=None):
    if unit_eps is None:
        unit_eps = translate.theta2chord(np.pi/32)
    lum = lf.apply_coef(1)
    wv = np.einsum('ij,i->ij', lf.vec, lum)
    c = _cluster(wv, unit_eps * np.max(lum))
    ovec = []
    olum = []
    ooga = []
