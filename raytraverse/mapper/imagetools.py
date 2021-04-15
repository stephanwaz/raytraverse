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


def hdr2vol(imgf):
    ar = io.hdr2array(imgf)
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
