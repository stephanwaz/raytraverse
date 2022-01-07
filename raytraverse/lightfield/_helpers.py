# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

"""parallelization functions for integration"""
import numpy as np

from raytraverse import io
from raytraverse.lightpoint import LightPointKD


def evaluate_pt(skpoint, snpoint, skyvecs, suns, dproxy, vm=None, vms=None,
                 metricclass=None, metrics=None, srconly=False,
                 sumsafe=False, **kwargs):
    """point by point evaluation suitable for submitting to ProcessPool"""
    if srconly:
        sunskypt = [snpoint]
        smtx = [suns[:, 3]]
    elif sumsafe:
        sunskypt = [skpoint, snpoint]
        smtx = [skyvecs, suns[:, 3]]
    else:
        sunskypt = [skpoint.add(snpoint)]
        smtx = [np.hstack((skyvecs, suns[:, 3:4]))]
    if len(vms) == 1:
        args = (vms[0].dxyz, vms[0].viewangle * vms[0].aspect)
        didx = [lpt.query_ball(*args)[0] for lpt in sunskypt]
    else:
        didx = [None] * len(sunskypt)
    srcs = []
    for lpt, di, sx in zip(sunskypt, didx, smtx):
        pts = []
        for skyvec, sun in zip(sx, suns):
            vol = lpt.evaluate(skyvec, vm=vm, idx=di,
                               srcvecoverride=sun[0:3], srconly=srconly)
            views = []
            for v in vms:
                views.append(metricclass(*vol, v, metricset=metrics,
                                         **kwargs)())
            views = np.stack(views)
            pts.append(views)
        srcs.append(np.stack(pts))
    return np.sum(srcs, axis=0)


def img_pt(skpoint, snpoint, skyvecs, suns, dproxy, vms=None,  combos=None,
            qpts=None, skinfo=None, res=512, interp=False, prefix="img"):
    """point by point evaluation suitable for submitting to ProcessPool"""
    outfs = []
    lpinfo = ["SUNPOINT= loc: ({:.3f}, {:.3f}, {:.3f}) src: ({:.3f}, {:.3f}, "
              "{:.3f}) {}".format(*snpoint.pt, *snpoint.srcdir[0],
                                  snpoint.file)]
    if skpoint is not None:
        lpinfo.append("SKYPOINT= loc: ({:.3f}, {:.3f}, "
                      "{:.3f}) {}".format(*skpoint.pt, skpoint.file))
    sky_i = -1
    for i, v in enumerate(vms):
        img, pdirs, mask, mask2, header = v.init_img(res)
        if interp:
            sun_i = None
            sky_i = None
        else:
            sun_i, _ = snpoint.query_ray(pdirs[mask])
            if skpoint is not None:
                sky_i, _ = skpoint.query_ray(pdirs[mask])
        for skyvec, sun, c, info, qpt in zip(skyvecs, suns, combos[:, i],
                                             skinfo, qpts):
            header = [v.header(qpt), "SKYCOND= sunpos: ({:.3f}, {:.3f}, {:.3f})"
                      " dirnorm: {} diffhoriz: {}".format(*info)] + lpinfo
            if skpoint is not None:
                skpoint.add_to_img(img, pdirs[mask], mask, vm=v, interp=interp,
                                   idx=sky_i, skyvec=skyvec)
            snpoint.add_to_img(img, pdirs[mask], mask, vm=v, interp=interp,
                               idx=sun_i, skyvec=[sun[3]])
            outf = "{}_{}_{}_{}.hdr".format(prefix, *c)
            outfs.append(outf)
            io.array2hdr(img, outf, header)
            img[:] = 0
    return outfs


def indirect_to_suns(snp, skp, skd, skpatch, omegar, scene, srcprefix="i_"):
    src = f"{srcprefix}{snp.src}"
    pf1 = f"{scene.outdir}/{skp.parent}/{snp.src}_points.tsv"
    pf2 = f"{scene.outdir}/{skp.parent}/{src}_points.tsv"
    skvec = skp.vec
    sklum = np.maximum((skp.lum - skd.lum)[:, skpatch]*omegar, 0)
    ski = LightPointKD(scene, vec=skvec, lum=sklum, vm=skp.vm,
                       pt=skp.pt, posidx=skp.posidx, src='indirect',
                       srcn=1, srcdir=skp.srcdir[skpatch],
                       write=False, omega=skp.omega, parent=skp.parent)
    snp.add(ski, src=src, calcomega=True, write=True, sumsrc=True)
    return pf1, pf2
