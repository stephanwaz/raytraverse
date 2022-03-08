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
                sumsafe=False, suntol=1.0, svengine=None, blursun=False,
                refl=None, resamprad=0.0, **kwargs):
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
    resampi, resampvecs = prep_resamp(sunskypt, refl, resamprad)
    # loop over lightpoints to sum
    for ri, (lpt, di, sx) in enumerate(zip(sunskypt, didx, smtx)):
        pts = []
        t_srcview = (lpt.srcviews, lpt.srcviewidxs)
        # loop over skyvecs to evaluate
        for skyvec, sun in zip(sx, suns):
            if svengine is not None and ri == resampi:
                update_src_view(svengine, lpt, sun[0:3], vm, suntol, refl=refl,
                                resampvecs=resampvecs, resamprad=resamprad)
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
           suntol=1.0, svengine=None, refl=None, resamprad=0.0, **kwargs):
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
    resampi, resampvecs = prep_resamp(lpts, refl, resamprad)
    for j, (c, info, qpt, sun) in enumerate(zip(combos, skinfo, qpts, suns)):
        for i, v, pdirs, mask, lp_is, w in vinfos:
            header = [v.header(qpt), "SKYCOND= sunpos: ({:.3f}, {:.3f}, {:.3f})"
                      " dirnorm: {} diffhoriz: {}".format(*info)] + lpinfo
            for ri, (lp_i, lp, svec) in enumerate(zip(lp_is, lpts, skyvecs)):
                if svengine is not None and ri == resampi:
                    update_src_view(svengine, lp, sun[0:3], v, suntol,
                                    refl=refl, resampvecs=resampvecs, resamprad=resamprad)
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


def prep_resamp(lpts, refl=None, resamprad=0.0):
    sunpt = [i for i, s in enumerate(lpts) if "_sun_" in s.src]
    sunpt2 = [i for i, s in enumerate(lpts) if "sun" in s.src]
    if len(sunpt) > 0:
        resampi = sunpt[0]
    elif len(sunpt2) > 0:
        resampi = sunpt2[0]
    else:
        resampi = -1
    srcdir = lpts[resampi].srcdir[-1:]
    if refl is not None:
        refl = translate.norm(refl.reshape(-1, 3))
        sunr = translate.reflect(srcdir, refl, False)
        srcdir = np.vstack((srcdir, sunr[0]))
    if resampi >= 0 and resamprad > 0.0:
        resampvecs = []
        for src in srcdir:
            i = lpts[resampi].query_ball(src, resamprad)
            resampvecs.append(i[0])
    else:
        resampvecs = None
    return resampi, resampvecs


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


def _in_view(vm, suna, sunb, tol=0.533):
    return (vm.degrees(suna)[0] <= (vm.viewangle + tol)/2 or
            vm.degrees(sunb)[0] <= (vm.viewangle + tol)/2)


def update_src_view(engine, lpt, sun, vm=None, tol=1.0, refl=None,
                    resampvecs=None, reflarea=None, resamprad=0.0):
    rtol = max(0.533, resamprad)
    if translate.degrees(sun, lpt.srcdir)[0] <= tol:
        return None
    if reflarea is None:
        reflarea = [0.533]
    if vm is None:
        vm = ViewMapper()
    inview = [_in_view(vm, sun, lpt.srcdir[-1], rtol)]
    svm = [ViewMapper(sun, reflarea[0], "sunview")]
    srcdir = [sun]
    if refl is not None and len(refl) > 0:
        if len(reflarea) < 1 + len(refl):
            reflarea += [0.533]*len(refl)
        refl = translate.norm(refl.reshape(-1, 3))
        sunr = translate.reflect(sun.reshape(1, 3), refl, False)
        for i, (sr, m, ar) in enumerate(zip(*sunr, reflarea[1:])):
            if m and ar > 0:
                rsrc = translate.reflect(lpt.srcdir[-1:], refl)[0][0]
                srcdir.append(rsrc)
                vm = ViewMapper(sr, ar, f"reflection_{i:03d}")
                svm.append(vm)
                inview.append(_in_view(vm, sr, rsrc, rtol))
    dosamp = np.any(inview)
    if len(svm) > 0 and dosamp:
        viewsampler = SunSamplerPtView(lpt.scene, engine, sun, 0)
        svpoints = viewsampler.run(lpt.pt, 0, vm=svm)

        lpt.set_srcviews(svpoints)
        if resampvecs is not None and dosamp:
            vecs = []
            ri = []
            for sd, rs, iv, sv in zip(srcdir, resampvecs, inview, svm):
                if inview and len(rs) > 0:
                    ymtx, pmtx = translate.rmtx_yp(np.stack((sd, sv.dxyz)))
                    # rotate to z-up in lp.srcdir space
                    vec = np.einsum("ij,kj,li->kl", ymtx[0], lpt.vec[rs], pmtx[0])
                    # rotate back in actual sun space
                    vec = np.einsum("ji,kj,il->kl", pmtx[1], vec, ymtx[1])
                    lucky_squirel = sv.degrees(vec) > 0.533/2
                    if np.sum(lucky_squirel) > 0:
                        vecs.append(vec[lucky_squirel])
                        ri.append(np.array(rs)[lucky_squirel])
            if len(ri) > 0:
                rvecs = np.vstack(vecs)
                ris = np.concatenate(ri)
                rvecs = np.concatenate(np.broadcast_arrays(lpt.pt[None, :],
                                                           rvecs), 1)
                lpt.lum[ris, -1] = engine(rvecs).ravel()
