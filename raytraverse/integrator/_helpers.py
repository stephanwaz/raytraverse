# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

"""parallelization functions for integration"""
import numpy as np
from scipy.spatial import Voronoi
from shapely.geometry import Polygon

from raytraverse import io, translate
from raytraverse.mapper import ViewMapper
from raytraverse.sampler import SunSamplerPtView


def evaluate_pt(lpts, skyvecs, suns, vm=None, vms=None,
                metricclass=None, metrics=None, srconly=False,
                sumsafe=False, suntol=10.0, svengine=None, blursun=False,
                refl=None, reflarea=None, **kwargs):
    """point by point evaluation suitable for submitting to ProcessPool"""
    if len(lpts) == 0:
        return np.zeros((len(suns), len(vms), len(metrics)))
    if srconly or sumsafe:
        sunskypt = lpts
        smtx = skyvecs
    else:
        lp0 = lpts[0]
        for lp in lpts[1:]:
            lp0 = lp0.add(lp)
        sunskypt = [lp0]
        smtx = [np.hstack(skyvecs)]

    if len(vms) == 1:
        args = (vms[0].dxyz, vms[0].viewangle*vms[0].aspect)
        didx = [lpt.query_ball(*args)[0] for lpt in sunskypt]
    else:
        didx = [None]*len(sunskypt)
    srcs = []
    sunpt = [i for i, s in enumerate(sunskypt) if "_sun_" in s.src]
    sunpt2 = [i for i, s in enumerate(sunskypt) if "sun" in s.src]
    if len(sunpt) > 0:
        resampi = sunpt[0]
    elif len(sunpt2) > 0:
        resampi = sunpt2[0]
    else:
        resampi = -1
    # loop over lightpoints to sum
    for ri, (lpt, di, sx) in enumerate(zip(sunskypt, didx, smtx)):
        pts = []
        t_srcview = (lpt.srcviews, lpt.srcviewidxs)
        # loop over skyvecs to evaluate
        for skyvec, sun in zip(sx, suns):
            if svengine is not None and ri == resampi:
                update_src_view(svengine, lpt, sun[0:3], vm, suntol, refl=refl, reflarea=reflarea)
            vol = lpt.evaluate(skyvec, vm=vm, idx=di, blursun=blursun,
                               srconly=srconly)
            views = []
            for v in vms:
                views.append(metricclass(*vol, v, metricset=metrics,
                                         **kwargs)())
            views = np.stack(views)
            pts.append(views)
            lpt.set_srcviews(*t_srcview)
        srcs.append(np.stack(pts))
    return np.sum(srcs, axis=0)


def img_pt(lpts, skyvecs, suns, vms=None,  combos=None,
           qpts=None, skinfo=None, res=512, interp=False, prefix="img",
           suntol=10.0, svengine=None, refl=None, reflarea=None, **kwargs):
    """point by point evaluation suitable for submitting to ProcessPool"""
    outfs = []
    lpinfo = ["LPOINT{}= loc: ({:.3f}, {:.3f}, {:.3f}) src: ({:.3f}, {:.3f}, "
              "{:.3f}) {}".format(i, *lpt.pt, *lpt.srcdir[0], lpt.file)
              for i, lpt in enumerate(lpts)]
    if interp:
        lp0 = lpts[0]
        for lp in lpts[1:]:
            lp0 = lp0.add(lp)
        lpts = [lp0]
        skyvecs = [np.hstack(skyvecs)]

    vinfos = []
    for i, v in enumerate(vms):
        img, pdirs, mask, mask2, header = v.init_img(res)
        if interp == "high":
            lp_is, w = lpts[0].content_interp_wedge(svengine, pdirs[mask],
                                                    bandwidth=10, **kwargs)
            lp_is = (lp_is,)
        elif interp == "fast":
            lp_is, w = lpts[0].content_interp(svengine, pdirs[mask], **kwargs)
            lp_is = (lp_is,)
        elif interp:
            lp_is = [None] * len(lpts)
            w = None
        else:
            lp_is = [lpt.query_ray(pdirs[mask])[0] for lpt in lpts]
            w = None
        vinfos.append((i, v, pdirs, mask, lp_is, w))
    if interp in ['high', 'fast']:
        interp = 'precomp'
    sunpt = [i for i, s in enumerate(lpts) if "_sun_" in s.src]
    sunpt2 = [i for i, s in enumerate(lpts) if "sun" in s.src]
    if len(sunpt) > 0:
        resampi = sunpt[0]
    elif len(sunpt2) > 0:
        resampi = sunpt2[0]
    else:
        resampi = -1
    for j, (c, info, qpt, sun) in enumerate(zip(combos, skinfo, qpts, suns)):
        for i, v, pdirs, mask, lp_is, w in vinfos:
            header = [v.header(qpt), "SKYCOND= sunpos: ({:.3f}, {:.3f}, {:.3f})"
                      " dirnorm: {} diffhoriz: {}".format(*info)] + lpinfo
            for ri, (lp_i, lp, svec) in enumerate(zip(lp_is, lpts, skyvecs)):
                if svengine is not None and ri == resampi:
                    update_src_view(svengine, lp, sun[0:3], v, suntol, refl=refl, reflarea=reflarea)
                lp.add_to_img(img, pdirs[mask], mask, vm=v, interp=interp,
                              idx=lp_i, skyvec=svec[j], engine=svengine,
                              interpweights=w)
                if np.min(svec) < 0:
                    img = np.maximum(img, 0)
            outf = "{}_{}_{}_{}.hdr".format(prefix, *c[i])
            outfs.append(outf)
            io.array2hdr(img, outf, header)
            img[:] = 0
    return outfs


def calc_omega(vecs, pm):
    """calculate area"""
    # border capture any infinite edges
    bordered = np.concatenate((vecs,
                               pm.bbox_vertices(pm.area**.5 * 10)))
    vor = Voronoi(bordered[:, 0:2])
    omega = []
    for i in range(len(vecs)):
        region = vor.regions[vor.point_region[i]]
        p = Polygon(vor.vertices[region])
        area = 0
        for bord in pm.borders():
            mask = Polygon(bord)
            area += p.intersection(mask).area
        omega.append(area)
    return np.asarray(omega)


def update_src_view(engine, lpt, sun, vm=None, tol=10.0, refl=None, reflarea=None):
    if reflarea is None:
        reflarea = [0.533]
    if vm is None:
        vm = ViewMapper()
    if (translate.degrees(sun, lpt.srcdir)[0] <= tol or
            vm.degrees(sun)[0] > vm.viewangle/2):
        svm = []
    else:
        svm = [ViewMapper(sun, reflarea[0], "sunview")]
    if refl is None:
        try:
            refl = np.loadtxt(f"{lpt.scene.outdir}/{lpt.parent}/"
                              f"reflection_normals.txt")
            if len(reflarea) < 1 + len(refl):
                reflarea += [1.066] * len(refl)
        except OSError:
            pass
    if refl is not None and len(refl) > 0:
        refl = translate.norm(refl.reshape(-1, 3))
        sunr = translate.reflect(sun.reshape(1, 3), refl, False)
        for i, (sr, m, ar) in enumerate(zip(*sunr, reflarea[1:])):
            if m and ar > 0:
                svm.append(ViewMapper(sr, ar, f"reflection_{i:03d}"))
    if len(svm) > 0:
        viewsampler = SunSamplerPtView(lpt.scene, engine, sun, 0)
        svpoints = viewsampler.run(lpt.pt, 0, vm=svm)
        lpt.set_srcviews(svpoints)
