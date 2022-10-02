# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

"""parallelization functions for integration"""
import numpy as np

from raytraverse import io, translate
from raytraverse.mapper import ViewMapper
from raytraverse.sampler import SunSamplerPtView
from raytraverse.lightpoint import LightPointKD


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
        img, pdirs, mask, mask2, header = v.init_img(res, jitter=.5,
                                                     features=lpts[0].features)
        if interp == "highc":
            lp_is, w = lpts[0].interp(pdirs[mask], rt=svengine, **kwargs)
            lp_is = (lp_is,)
        elif interp == "high":
            lp_is, w = lpts[0].interp(pdirs[mask], **kwargs)
            lp_is = (lp_is,)
        elif interp == "fastc":
            lp_is, w = lpts[0].interp(pdirs[mask], angle=False, lum=False,
                                      dither=True, rt=svengine,
                                      **kwargs)
            lp_is = (lp_is,)
        elif interp == "fast":
            lp_is, w = lpts[0].interp(pdirs[mask], angle=False, lum=False,
                                      dither=True, **kwargs)
            lp_is = (lp_is,)
        elif interp:
            lp_is = [None] * len(lpts)
            w = None
        else:
            lp_is = [lpt.query_ray(pdirs[mask])[0] for lpt in lpts]
            w = None
        vinfos.append((i, v, img, pdirs, mask, mask2, lp_is, w))
    if interp in ['high', 'fast', 'highc', 'fastc']:
        interp = 'precomp'
    resampi, resampvecs = prep_resamp(lpts, refl, resamprad)
    for j, (c, info, qpt, sun) in enumerate(zip(combos, skinfo, qpts, suns)):
        for i, v, img, pdirs, mask, mask2, lp_is, w in vinfos:
            header = [v.header(qpt), "SKYCOND= sunpos: ({:.3f}, {:.3f}, {:.3f})"
                      " dirnorm: {} diffhoriz: {}".format(*info)] + lpinfo
            for ri, (lp_i, lp, svec) in enumerate(zip(lp_is, lpts, skyvecs)):
                if svengine is not None and ri == resampi:
                    update_src_view(svengine, lp, sun[0:3], v, suntol,
                                    refl=refl, resampvecs=resampvecs,
                                    resamprad=resamprad)
                lp.add_to_img(img, pdirs[mask], mask2, vm=v, interp=interp,
                              idx=lp_i, skyvec=svec[j], engine=svengine,
                              interpweights=w)
                if np.min(svec) < 0:
                    img = np.maximum(img, 0)
            outf = "{}_{}_{}_{}.hdr".format(prefix, *c[i])
            outfs.append(outf)
            io.array2hdr(img, outf, header)
            img[:] = 0
    return outfs


def prep_ds(lpts, skyvecs):
    try:
        snp = lpts[2]
    except IndexError:
        # this implies the sunpt is not needed (0 direct solar)
        return lpts[0:1], skyvecs[0:1], None
    skyvecs = [np.hstack(skyvecs[0:2]), skyvecs[2]]
    ski, skyvecs, refl = apply_dsky_patch(lpts[0], lpts[1], skyvecs, snp.srcdir)
    lpts = [ski, snp]
    return lpts, skyvecs, refl


def evaluate_pt_ds(lpts, skyvecs, suns, **kwargs):
    lpts, skyvecs, refl = prep_ds(lpts, skyvecs)
    return evaluate_pt(lpts, skyvecs, suns, refl=refl, **kwargs)


def img_pt_ds(lpts, skyvecs, suns, **kwargs):
    lpts, skyvecs, refl = prep_ds(lpts, skyvecs)
    return img_pt(lpts, skyvecs, suns, refl=refl, **kwargs)


def _prep_dv(lpts, skyvecs, skdir):
    dirlum = np.zeros((len(lpts[0].lum), 1))
    ski, skyvecs, refl = apply_dsky_patch(lpts[0], lpts[1], skyvecs, skdir,
                                          dirlum)
    lpts = [ski]
    return lpts, skyvecs, refl


def evaluate_pt_dv(lpts, skyvecs, suns, **kwargs):
    side = int((skyvecs[0].shape[1] - 1)**.5)
    skpatch = translate.xyz2skybin(suns[0], side)[0]
    skdir = translate.skybin2xyz(skpatch, side)[0]
    lpts, skyvecs, refl = _prep_dv(lpts, skyvecs, skdir)
    kwargs.update(suntol=-1)
    return evaluate_pt(lpts, skyvecs, suns, refl=refl, **kwargs)


def img_pt_dv(lpts, skyvecs, suns, **kwargs):
    side = int((skyvecs[0].shape[1] - 1)**.5)
    skpatch = translate.xyz2skybin(suns[0], side)[0]
    skdir = translate.skybin2xyz(skpatch, side)[0]
    lpts, skyvecs, refl = _prep_dv(lpts, skyvecs, skdir)
    kwargs.update(suntol=-1)
    return img_pt(lpts, skyvecs, suns, refl=refl, **kwargs)


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
                    lucky_squirel = sv.degrees(vec) > 0.534/2
                    if np.sum(lucky_squirel) > 0:
                        vecs.append(vec[lucky_squirel])
                        ri.append(np.array(rs)[lucky_squirel])
            if len(ri) > 0:
                rvecs = np.vstack(vecs)
                ris = np.concatenate(ri)
                rvecs = np.concatenate(np.broadcast_arrays(lpt.pt[None, :],
                                                           rvecs), 1)
                lpt.lum[ris, -1] = engine(rvecs).ravel()


def apply_dsky_patch(skp, skd, skyvecs, skdir, dirlum=None):
    skpatch = translate.xyz2skybin(skdir, int((skp.srcn - 1)**.5))[0]
    skvec = skp.vec
    skydlum = np.copy(skd.lum[:, skpatch])
    sklum = np.maximum((skp.lum[:, skpatch] - skydlum), 0)[:, None]
    sklum = np.hstack((skp.lum, sklum))

    srcn = skp.srcn + 1
    if len(skyvecs[0]) < sklum.shape[1]:
        sklum = np.einsum('ij,kj->ki', skyvecs[0], sklum)
        srcn = len(skyvecs[0])
        skyvecs[0] = np.eye(len(skyvecs[0]))

    if dirlum is not None:
        srcn += 1
        sklum = np.hstack((sklum, dirlum))
        skyvecs = [np.hstack(skyvecs)]

    ski = LightPointKD(skp.scene, vec=skvec, lum=sklum, vm=skp.vm,
                       pt=skp.pt, posidx=skp.posidx,
                       src=f'sky_isun{skpatch:04d}', srcn=srcn,
                       srcdir=skdir,
                       write=False, omega=skp.omega, parent=skp.parent)
    try:
        refl = io.load_txt(f"{skp.scene.outdir}/{skp.parent}/"
                           f"reflection_normals.txt")
    except FileNotFoundError:
        refl = None
    else:
        refl = translate.norm(refl.reshape(-1, 3))
    return ski, skyvecs, refl
