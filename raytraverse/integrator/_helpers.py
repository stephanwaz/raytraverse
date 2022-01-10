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


def evaluate_pt(lpts, skyvecs, suns, vm=None, vms=None,
                metricclass=None, metrics=None, srconly=False,
                sumsafe=False, **kwargs):
    """point by point evaluation suitable for submitting to ProcessPool"""
    if srconly or sumsafe:
        sunskypt = lpts
        smtx = skyvecs
    else:
        lp0 = lpts[0]
        for lp in lpts[1:]:
            lp0 = lp0.add(lp)
        sunskypt = [lp0]
        print(lp0.srcn)
        smtx = [np.hstack(skyvecs)]

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


def img_pt(lpts, skyvecs, suns, vms=None,  combos=None,
           qpts=None, skinfo=None, res=512, interp=False, prefix="img"):
    """point by point evaluation suitable for submitting to ProcessPool"""
    outfs = []
    lpinfo = ["LPOINT{}= loc: ({:.3f}, {:.3f}, {:.3f}) src: ({:.3f}, {:.3f}, "
              "{:.3f}) {}".format(i, *lpt.pt, *lpt.srcdir[0], lpt.file)
              for i, lpt in enumerate(lpts)]
    for i, v in enumerate(vms):
        img, pdirs, mask, mask2, header = v.init_img(res)
        if interp:
            lp_is = [None] * len(lpts)
        else:
            lp_is = [lpt.query_ray(pdirs[mask])[0] for lpt in lpts]
        for j, (c, info, qpt) in enumerate(zip(combos[:, i], skinfo, qpts)):
            header = [v.header(qpt), "SKYCOND= sunpos: ({:.3f}, {:.3f}, {:.3f})"
                      " dirnorm: {} diffhoriz: {}".format(*info)] + lpinfo
            for lp_i, lp, svec in zip(lp_is, lpts, skyvecs):
                lp.add_to_img(img, pdirs[mask], mask, vm=v, interp=interp,
                              idx=lp_i, skyvec=svec[j])
                if np.min(svec) < 0:
                    img = np.maximum(img, 0)
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
