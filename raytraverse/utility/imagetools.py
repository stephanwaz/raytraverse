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
from raytraverse.evaluate import MetricSet
from raytraverse.mapper.viewmapper import ViewMapper


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


def hdr2vol(imgf, vm=None):
    ar = io.hdr2array(imgf).T
    if vm is None:
        vm = hdr2vm(imgf)
    vecs = vm.pixelrays(ar.shape[-1]).reshape(-1, 3)
    oga = vm.pixel2omega(vm.pixels(ar.shape[-1]), ar.shape[-1]).ravel()
    return vecs, oga, ar.ravel()


def hdr2vm(imgf):
    header = cst.pipeline([f"getinfo {imgf}"])
    if "VIEW= -vta" in header:
        vp = header.rsplit("VIEW= -vta", 1)[-1].splitlines()[0].split()
        view_angle = float(vp[vp.index("-vh") + 1])
        vd = vp.index("-vd")
        view_dir = [float(vp[i]) for i in range(vd + 1, vd + 4)]
        hd = cst.pipeline([f"getinfo -d {imgf}"]).strip().split()
        x = 1
        y = 1
        for i in range(2, len(hd)):
            if 'X' in hd[i - 1]:
                x = float(hd[i])
            elif 'Y' in hd[i - 1]:
                y = float(hd[i])
        return ViewMapper(view_dir, view_angle * x / y)
    else:
        return None


def normalize_peak(v, o, l, scale=179, peaka=6.7967e-05, peakt=1e5, peakr=4):
    pc = np.nonzero(l > peakt / scale)[0]
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
        # treat as full sun
        if cume[-1] > esun:
            stop = np.argmax(cume > esun)
            if stop == 0:
                stop = len(cume)
            peakl = cume[stop - 1]/peaka
        # treat as partial sun (needs to use peak ratio)
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
        v = np.vstack((vol[:, 0:3], pv))
        o = np.concatenate((vol[:, 3], [peaka]))
        l = np.concatenate((vol[:, 4], [peakl]))
    return v, o, l


def imgmetric(imgf, metrics, peakn=False, scale=179, threshold=2000.,
              **peakwargs):
    vm = hdr2vm(imgf)
    if vm is None:
        vm = ViewMapper(viewangle=180)
    v, o, l = hdr2vol(imgf, vm)
    if peakn:
        v, o, l = normalize_peak(v, o, l, scale, **peakwargs)
    return MetricSet(v, o, l, vm, metrics, scale=scale, threshold=threshold)()
