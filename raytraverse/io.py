# -*- coding: utf-8 -*-

# Copyright (c) 2019 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

"""functions for reading and writing"""
from datetime import datetime, timezone
import shlex
import os
import sys
from subprocess import Popen, PIPE
from scipy.ndimage.filters import uniform_filter

import numpy as np

import raytraverse
from raytraverse import translate
from raytraverse.mapper import ViewMapper
from raytraverse.crenderer import cRtrace


def get_nproc(nproc=None):
    if nproc is not None:
        return nproc
    env_nproc = os.getenv('RAYTRAVERSE_PROC_CAP')
    try:
        return int(env_nproc)
    except (ValueError, TypeError):
        return os.cpu_count()


def set_nproc(nproc):
    if nproc is None:
        return None
    if type(nproc) != int:
        raise ValueError('nproc must be an int')
    if nproc < 1:
        unset_nproc()
    else:
        os.environ['RAYTRAVERSE_PROC_CAP'] = str(nproc)


def unset_nproc():
    try:
        os.environ.pop('RAYTRAVERSE_PROC_CAP')
    except KeyError:
        pass


def np2bytes(ar, dtype='<f'):
    """format ar as bytestring

    Parameters
    ----------
    ar: np.array
    dtype: str
        argument to pass to np.dtype()

    Returns
    -------
    bytes
    """
    dt = np.dtype(dtype)
    return ar.astype(dt).tobytes()


def np2bytefile(ar, outf, dtype='<f', mode='wb'):
    """save vectors to file

    Parameters
    ----------
    ar: np.array
        array to write
    outf: str
        file to write to
    dtype: str
        argument to pass to np.dtype()
    """
    f = open(outf, mode)
    f.write(np2bytes(ar, dtype=dtype))
    f.close()


def bytes2np(buf, shape, dtype='<f'):
    """read ar from bytestring

    Parameters
    ----------
    buf: bytes, str
    shape: tuple
        array shape
    dtype: str
        argument to pass to np.dtype()

    Returns
    -------
    np.array
    """
    dt = np.dtype(dtype)
    return np.frombuffer(buf, dtype=dt).reshape(*shape)


def bytefile2np(f, shape, dtype='<f'):
    """read binary data from f

    Parameters
    ----------
    f: IOBase
        file object to read array from
    shape: tuple
        array shape
    dtype: str
        argument to pass to np.dtype()

    Returns
    -------
    ar.shape
        necessary for reconstruction
    """
    return bytes2np(f.read(), shape, dtype)


def version_header():
    """generate image header string"""
    lastmod = os.path.getmtime(os.path.dirname(raytraverse.__file__))
    lm = datetime.fromtimestamp(lastmod).strftime("%Y-%m-%d")
    cap = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
    return ["CAPDATE= " + cap,
            f"SOFTWARE= RAYTRAVERSE {raytraverse.__version__} lastmod {lm} //"
            f" {cRtrace.version}"]


def _array2hdr(ar, imgf, header, pval):
    """write 2d np.array to hdr image format

        Parameters
        ----------
        ar: np.array
            image array
        imgf: str
            file path to right
        header: list
            list of header lines to append to image header
        pval: str
            pvalue command

        Returns
        -------
        imgf
        """
    if imgf is None:
        f = None
    else:
        f = open(imgf, 'wb')
    if header is not None:
        hdr = "' '".join(header)
        getinfo = shlex.split(f"getinfo -a '{hdr}'")
        p = Popen(pval.split(), stdin=PIPE, stdout=PIPE)
        q = Popen(getinfo, stdin=p.stdout, stdout=f)
    else:
        p = Popen(pval.split(), stdin=PIPE, stdout=f)
        q = p
    p.stdin.write(np2bytes(ar))
    p.stdin.flush()
    q.communicate()
    try:
        f.close()
    except AttributeError:
        pass
    return imgf


def array2hdr(ar, imgf, header=None):
    """write 2d np.array (x,y) to hdr image format

    Parameters
    ----------
    ar: np.array
            image array
    imgf: str
        file path to right
    header: list
        list of header lines to append to image header

    Returns
    -------
    imgf
    """
    pval = f'pvalue -r -b -h -H -df -o -y {ar.shape[1]} +x {ar.shape[0]}'
    return _array2hdr(ar[-1::-1, -1::-1].T, imgf, header, pval)


def carray2hdr(ar, imgf, header=None):
    """write color channel np.array (3, x, y) to hdr image format

    Parameters
    ----------
    ar: np.array
            image array
    imgf: str
        file path to right
    header: list
        list of header lines to append to image header

    Returns
    -------
    imgf
    """
    pval = f'pvalue -r -h -H -df -o -y {ar.shape[-1]} +x {ar.shape[-2]}'
    return _array2hdr(ar.T[-1::-1, -1::-1, :], imgf, header, pval)


def uvarray2hdr(uvarray, imgf, header=None):
    res = uvarray.shape[0]
    vm = ViewMapper(viewangle=180)
    pixelxyz = vm.pixelrays(res)
    uv = vm.xyz2uv(pixelxyz.reshape(-1, 3))
    mask = vm.in_view(pixelxyz, indices=False)
    ij = translate.uv2ij(uv[mask], res)
    img = np.zeros(res*res)
    img[mask] = uvarray[ij[:, 0], ij[-1:None:-1, 1]]
    array2hdr(img.reshape(res, res), imgf, header)


def hdr2array(imgf):
    """read np.array from hdr image

    Parameters
    ----------
    imgf: file path of image

    Returns
    -------
    ar: np.array

    """
    pval = f'pvalue -b -h -df -o {imgf}'
    p = Popen(pval.split(), stdout=PIPE)
    shape = p.stdout.readline().strip().split()
    shape = (int(shape[-3]), int(shape[-1]))
    return bytes2np(p.stdout.read(), shape)[-1::-1, -1::-1]


def rgb2rad(rgb):
    try:
        return np.einsum('ij,j', rgb, [0.265, 0.670, 0.065])
    except ValueError:
        return np.einsum('j,j', rgb, [0.265, 0.670, 0.065])


def rgb2lum(rgb):
    return np.einsum('ij,j', rgb, [47.435, 119.93, 11.635])


def rgbe2lum(rgbe):
    """
    convert from Radiance hdr rgbe 4-byte data format to floating point
    luminance.

    Parameters
    ----------
    rgbe: np.array
        r,g,b,e unsigned integers according to:
        http://radsite.lbl.gov/radiance/refer/filefmts.pdf

    Returns
    -------
    lum: luminance in cd/m^2
    """
    v = np.power(2., rgbe[:, 3] - 128).reshape(-1, 1) / 256
    rgb = np.where(rgbe[:, 0:3] == 0, 0, (rgbe[:, 0:3] + 0.5) * v)
    # luminance = 179 * (0.265*R + 0.670*G + 0.065*B)
    return rgb2lum(rgb)


def add_vecs_to_img(vm, img, v, channels=(1, 0, 0), grow=0):
    res = img.shape[-1]
    if vm.viewangle == 360:
        reverse = vm.degrees(v) > 90
        pa = vm.ivm.ray2pixel(v[reverse], res)
        pa[:, 0] += res
        pb = vm.ray2pixel(v[np.logical_not(reverse)], res)
        xp = np.concatenate((pa[:, 0], pb[:, 0]))
        yp = np.concatenate((pa[:, 1], pb[:, 1]))
    else:
        pb = vm.ray2pixel(v, res)
        xp = pb[:, 0]
        yp = pb[:, 1]
    r = int(grow*2 + 1)
    if len(img.shape) == 2:
        try:
            channel = channels[0]
        except TypeError:
            channel = channels
        img[xp, yp] = channel
        if grow > 0:
            img = uniform_filter(img*r**2, r)
    else:
        for i in range(img.shape[0]):
            if channels[i] is not None:
                img[i, xp, yp] = channels[i]
        if grow > 0:
            img = uniform_filter(img*r**2, (1, r, r))
    return img
