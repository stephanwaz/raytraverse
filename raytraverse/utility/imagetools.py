# -*- coding: utf-8 -*-

# Copyright (c) 2019 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

"""functions for translating from mappers to hdr"""
import numpy as np
import clasp.script_tools as cst


from raytraverse import translate, io
from raytraverse.evaluate import MetricSet, retina
from raytraverse.mapper.viewmapper import ViewMapper


def hdr_uv2ang(imgf, useview=None):
    imarray = io.hdr2carray(imgf)
    outf = imgf.rsplit(".", 1)[0] + "_ang.hdr"
    res = imarray.shape[-1]
    vm = ViewMapper(viewangle=180)
    pixelxyz = vm.pixelrays(res)
    uv = vm.xyz2uv(pixelxyz.reshape(-1, 3))
    mask = vm.in_view(pixelxyz, indices=False)
    ij = translate.uv2ij(uv[mask], res)
    img = np.zeros((3, res*res))
    img[:, mask] = imarray[:, ij[:, 1], ij[:, 0]]
    io.carray2hdr(img.reshape(3, res, res)[:, ::-1], outf)


def hdr_ang2uv(imgf, useview=True):
    vm = None
    if useview:
        vm = hdr2vm(imgf)
    if vm is None:
        vm = ViewMapper(viewangle=180)
    imarray = io.hdr2carray(imgf)
    outf = imgf.rsplit(".", 1)[0] + "_uv.hdr"
    res = imarray.shape[-1]
    uv = translate.bin2uv(np.arange(res*res), res)
    vm2 = ViewMapper(viewangle=180)
    xyz = vm.uv2xyz(uv)
    pxy = vm.ray2pixel(xyz, imarray.shape[-1])
    # img = np.take_along_axis(imarray, pxy.T, 0)
    img = imarray[:, pxy[:, 1], pxy[:, 0]].reshape(3, res, res)
    io.carray2hdr(img[:, ::-1], outf)


def uvarray2hdr(uvarray, imgf, header=None):
    res = uvarray.shape[0]
    vm = ViewMapper(viewangle=180)
    pixelxyz = vm.pixelrays(res)
    uv = vm.xyz2uv(pixelxyz.reshape(-1, 3))
    mask = vm.in_view(pixelxyz, indices=False)
    ij = translate.uv2ij(uv[mask], res)
    img = np.zeros(res*res)
    img[mask] = uvarray[ij[:, 0], ij[-1:None:-1, 1]]
    io.array2hdr(img.reshape(res, res), imgf, header)


def hdr2uvarray(imgf, vm=None, res=None):
    if vm is None:
        vm = hdr2vm(imgf)
    imarray = io.hdr2array(imgf)
    if res is None:
        res = imarray.shape[0]
    uv = translate.bin2uv(np.arange(res*res), res)
    xyz = vm.uv2xyz(uv)
    pxy = vm.ray2pixel(xyz, imarray.shape[0])
    return imarray[pxy[:, 1], pxy[:, 0]].reshape(res, res)


def hdr2vol(imgf, vm=None):
    ar = io.hdr2array(imgf).T
    if vm is None:
        vm = hdr2vm(imgf)
    vecs = vm.pixelrays(ar.shape[-1]).reshape(-1, 3)
    oga = vm.pixel2omega(vm.pixels(ar.shape[-1]), ar.shape[-1]).ravel()
    return vecs, oga, ar.ravel()


def vf_to_vm(view):
    """view file to ViewMapper"""
    vl = [i for i in open(view).readlines() if "-vta" in i]
    if len(vl) == 0:
        raise ValueError(f"no valid -vta view in file {view}")
    vp = vl[-1].split()
    view_angle = float(vp[vp.index("-vh") + 1])
    vd = vp.index("-vd")
    view_dir = [float(vp[i]) for i in range(vd + 1, vd + 4)]
    return ViewMapper(view_dir, view_angle)


def hdr2vm(imgf, vpt=False):
    """hdr to ViewMapper"""
    header, err = cst.pipeline([f"getinfo {imgf}"], caperr=True)
    try:
        err = err.decode("utf-8")
    except AttributeError:
        pass
    if "bad header!" in err:
        raise IOError(f"{err} - wrong file type?")
    elif "cannot open" in header:
        raise FileNotFoundError(f"{imgf} not found")
    vp = None
    if "VIEW= -vta" in header:
        vp = header.rsplit("VIEW= -vta", 1)[-1].splitlines()[0].split()
    elif "rvu -vta" in header:
        vp = header.rsplit("rvu -vta", 1)[-1].splitlines()[0].split()
    if vp is not None:
        view_angle = float(vp[vp.index("-vh") + 1])
        try:
            vd = vp.index("-vd")
        except ValueError:
            view_dir = [0.0, 1.0, 0.0]
        else:
            view_dir = [float(vp[i]) for i in range(vd + 1, vd + 4)]
        try:
            vpi = vp.index("-vp")
        except ValueError:
            view_pt = [0.0, 0.0, 0.0]
        else:
            view_pt = [float(vp[i]) for i in range(vpi + 1, vpi + 4)]
        hd = cst.pipeline([f"getinfo -d {imgf}"]).strip().split()
        x = 1
        y = 1
        for i in range(2, len(hd)):
            if 'X' in hd[i - 1]:
                x = float(hd[i])
            elif 'Y' in hd[i - 1]:
                y = float(hd[i])
        vm = ViewMapper(view_dir, view_angle * x / y)
    else:
        view_pt = None
        vm = None
    if vpt:
        return vm, view_pt
    else:
        return vm


def normalize_peak(v, o, l, scale=179, peaka=6.7967e-05, peakt=1e5, peakr=4,
                   blursun=False):
    """consolidate the brightest pixels represented by v, o, l up into a single
    source, correcting the area while maintaining equal energy

    Parameters
    ----------
    v: np.array
        shape (N, 3), direction vectors of pixels (x, y, z) normalized
    o: np.array
        shape (N,), solid angles of pixels (steradians)
    l: np.array
        shape (N,), luminance of pixels
    scale: Union[float, int], optional
        scale factor for l to convert to cd/m^2, default assumes radiance units
    peaka: float, optional
        area to aggregate to
    peakt: Union[float, int], optional
        lower threshold for possible bright pixels
    peakr: Union[float, int], optional
        ratio, from peak pixel value to lowest value to include when aggregating
        partially visible sun.
    blursun: bool, optional
        whether to correct area and luminance according to human PSF

    Returns
    -------
    v: np.array
        shape (N, 3), direction vectors of pixels (x, y, z) normalized
    o: np.array
        shape (N,), solid angles of pixels (steradians)
    l: np.array
        shape (N,), luminance of pixels

    """
    pc = np.nonzero(l > peakt / scale)[0]
    # only do something if some pixels above peakt
    if pc.size > 0:
        # first sort descending by luminance
        pc = pc[np.argsort(-l[pc])]
        pvol = np.hstack((v[pc], o[pc, None], l[pc, None]))
        # establish maximum radius for grouping
        cosrad = np.cos((peaka/np.pi)**.5*4)
        # calculate angular distance from peak ray and filter strays
        pd = np.einsum("i,ji->j", pvol[0, 0:3], pvol[:, 0:3])
        dm = pd > cosrad
        pc = pc[dm]
        pvol = pvol[dm]
        # calculate expected energy assuming full visibility:
        esun = pvol[0, 4]*peaka
        # sum up to peak energy
        cume = np.cumsum(pvol[:, 3]*pvol[:, 4])
        # when there is enough energy, treat as full sun
        if cume[-1] > esun:
            stop = np.argmax(cume > esun)
            if stop == 0:
                stop = len(cume)
            peakl = cume[stop - 1]/peaka
        # otherwise treat as partial sun (needs to use peak ratio)
        else:
            stop = np.argmax(pvol[:, 4] < pvol[0, 4]/peakr)
            if stop == 0:
                stop = len(cume)
            peakl = pvol[0, 4]
            peaka = cume[stop - 1]/peakl
        pc = pc[:stop]
        pvol = pvol[:stop]
        # new source vector weight by L*omega of source rarys
        pv = translate.norm(np.average(pvol[:, 0:3], axis=0,
                                       weights=pvol[:, 3]*pvol[:, 4]))
        # filter out source rays
        vol = np.delete(np.hstack((v, o[:, None], l[:, None])), pc, axis=0)
        # then add new consolidated ray back to output v, o, l
        v = np.vstack((vol[:, 0:3], pv))
        if blursun:
            cf = np.atleast_1d(retina.blur_sun(peaka, peakl))[0]
        else:
            cf = 1
        o = np.concatenate((vol[:, 3], [peaka*cf]))
        l = np.concatenate((vol[:, 4], [peakl/cf]))
    return v, o, l


def imgmetric(imgf, metrics, peakn=False, scale=179, threshold=2000., lowlight=False,
              **peakwargs):
    vm = hdr2vm(imgf)
    if vm is None:
        vm = ViewMapper(viewangle=180)
    v, o, l = hdr2vol(imgf, vm)
    if peakn:
        v, o, l = normalize_peak(v, o, l, scale, **peakwargs)
    return MetricSet(v, o, l, vm, metrics, scale=scale, threshold=threshold, lowlight=lowlight)()


# from raytraverse.lightpoint import LightPointKD
# from raytraverse.sampler import draw
# from raytraverse.sampler.basesampler import filterdict
# def img2lf(imga, imgb, src, scn):
#
#     accuracy = 0.5
#     t0 = 0
#     t1 = .6667
#     levels = 6
#
#     def _threshold(idx, acc):
#         """threshold for determining sample count"""
#         return acc * _linear(idx, t0, t1)
#
#     def _linear(x, x1, x2):
#         if levels <= 2:
#             return (x1, x2)[x]
#         else:
#             return (x2 - x1)/(levels - 1) * x + x1
#
#     vm, vp = hdr2vm(imga, vpt=True)
#     uva = hdr2uvarray(imga, vm, 1024)
#     if imgb is not None:
#         vmb = ViewMapper(vm.dxyz*np.array((-1, -1, -1)), vm.viewangle)
#         uvb = hdr2uvarray(imgb, vmb, 1024)
#         uva = np.stack((uvb, uva), 0).reshape(-1, uvb.shape[1])
#         vm = ViewMapper(vm.dxyz)
#     accuracy *= np.average(uva)
#     uvt = translate.resample(uva, uva.shape, radius=2)
#     ar = int(uva.shape[0]/uva.shape[1])
#     available = np.full(uvt.shape, True)
#     rays = None
#     vals = None
#     for i in range(1, levels+1):
#         res = 2**(levels-i)*1024/2**levels
#         uvt = translate.resample(uvt, (res*ar, res))
#         available = translate.resample(available.astype(float), uvt.shape) > 0.0
#         p = draw.get_detail(uvt, *filterdict['wav']).reshape(uvt.shape)
#         t = _threshold(levels-i, accuracy)
#         p[np.logical_not(available)] = 0
#         mi = p > t
#         available[mi] = False
#         miu = translate.resample(mi, (res*2*ar, res*2), False)
#         uv = vm.idx2uv(np.arange(miu.size)[miu.ravel()], miu.shape, False)
#         uv[:, 0] *= ar
#         ray = vm.uv2xyz(uv)
#         if rays is None:
#             rays = ray
#             vals = uva[miu]
#         else:
#             rays = np.concatenate((rays, ray))
#             vals = np.concatenate((vals, uva[miu]))
#         uva = translate.resample(uva, (res*ar, res))
#     lp = LightPointKD(scn, rays, vals, vm, vp, src=src)
#     lp.direct_view(512)



