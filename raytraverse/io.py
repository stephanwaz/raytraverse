# -*- coding: utf-8 -*-

# Copyright (c) 2019 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

"""functions for reading and writing"""
import shlex
import os
import sys
import threading
from subprocess import Popen, PIPE
from io import StringIO, BytesIO
from scipy.ndimage.filters import uniform_filter

import numpy as np

from raytraverse import translate
from raytraverse.mapper import ViewMapper


class CaptureStdOut:
    """redirect output streams at system level (including c printf)

    Parameters
    ----------

    b: bool, optional
        read data as bytes
    store: bool, optional
        record stdout in a IOStream, value accesible through self.stdout
    outf: IOBase, optional
        if not None, must be writable, closed on exit

    Notes
    -----
    ::

        with CaptureStdOut() as capture:
            do stuff
        capout = capture.stdout

    when using with pytest include the -s flag or this class has no effect

    """

    def __init__(self, b=False, store=True, outf=None):
        # Create pipe and dup2() the write end of it on top of stdout,
        # saving a copy of the old stdout
        if outf is not None and not hasattr(outf, "write"):
            raise AttributeError('If outf is not None, it must have a write'
                                 ' attribute')
        self.fileno = sys.stdout.fileno()
        self.save = os.dup(self.fileno)
        self.pipe = os.pipe()
        os.dup2(self.pipe[1], self.fileno)
        os.close(self.pipe[1])
        self._stdout = None
        self._file = outf
        if b:
            self.threader = threading.Thread(target=self.drain_bytes)
            if store:
                self._stdout = BytesIO()
        else:
            self.threader = threading.Thread(target=self.drain_str)
            if store:
                self._stdout = StringIO()
        self.threader.start()

    @property
    def stdout(self):
        try:
            return self._stdout.getvalue()
        except AttributeError:
            return None

    def __enter__(self):
        return self

    def _write(self, data):
        try:
            self._stdout.write(data)
        except AttributeError:
            pass
        try:
            self._file.write(data)
        except AttributeError:
            pass

    def drain_bytes(self):
        """read stdout as bytes"""
        while True:
            data = os.read(self.pipe[0], 1024)
            if not data:
                break
            self._write(data)

    def drain_str(self):
        """read stdout as unicode"""
        while True:
            data = os.read(self.pipe[0], 1024).decode()
            if not data:
                break
            self._write(data)

    def __exit__(self, exc_type, exc_value, traceback):
        """restore stdout and join threads"""
        # Close the write end of the pipe to unblock the reader thread and
        # trigger it to exit
        os.close(self.fileno)
        self.threader.join()
        # Clean up the pipe and restore the original stdout
        os.close(self.pipe[0])
        os.dup2(self.save, self.fileno)
        os.close(self.save)


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


def call_sampler(outf, command, vecs, shape):
    """make subprocess call to sampler given as command, expects rgb value
    as return for each vec

    Parameters
    ----------
    outf: str
        path to write out to
    command: str
        command line with executable and options
    vecs: np.array
        vectors to pass as stdin to command
    shape: tuple
        shape of expected output

    Returns
    -------
    lums: np.array
        of length vectors.shape[0]

    """
    f = open(outf, 'a+b')
    lum_file_pos = f.tell()
    p = Popen(shlex.split(command), stdout=f, stdin=PIPE)
    p.communicate(np2bytes(vecs))
    lum = bytefile2rad(f, shape, subs='ijk,k->ij', offset=lum_file_pos)
    return lum


def bytefile2rad(f, shape, slc=..., subs='ijk,k->ij', offset=0):
    memarray = np.memmap(f, dtype='<f', mode='r', shape=shape, offset=offset)
    return np.einsum(subs, memarray[slc], [0.265, 0.670, 0.065])


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


def array2hdr(ar, imgf, header=None):
    """write 2d np.array (x,y) to hdr image format

    Parameters
    ----------
    ar: np.array
    imgf: file path to right
    header: list of header lines to append to image header

    Returns
    -------

    """
    f = open(imgf, 'wb')
    pval = f'pvalue -r -b -h -H -df -o -y {ar.shape[1]} +x {ar.shape[0]}'
    if header is not None:
        hdr = "' '".join(header)
        getinfo = shlex.split(f"getinfo -a '{hdr}'")
        p = Popen(pval.split(), stdin=PIPE, stdout=PIPE)
        q = Popen(getinfo, stdin=p.stdout, stdout=f)
    else:
        p = Popen(pval.split(), stdin=PIPE, stdout=f)
        q = p
    p.stdin.write(np2bytes(ar[-1::-1, -1::-1].T))
    q.communicate()
    f.close()
    return imgf


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


def carray2hdr(ar, imgf, header=None):
    """write color channel np.array (3, x, y) to hdr image format

    Parameters
    ----------
    ar: np.array
    imgf: file path to right
    header: list of header lines to append to image header

    Returns
    -------

    """
    f = open(imgf, 'wb')
    pval = f'pvalue -r -h -H -df -o -y {ar.shape[-1]} +x {ar.shape[-2]}'
    if header is not None:
        hdr = "' '".join(header)
        getinfo = shlex.split(f"getinfo -a '{hdr}'")
        p = Popen(pval.split(), stdin=PIPE, stdout=PIPE)
        q = Popen(getinfo, stdin=p.stdout, stdout=f)
    else:
        p = Popen(pval.split(), stdin=PIPE, stdout=f)
        q = p
    p.stdin.write(np2bytes(ar.T[-1::-1, -1::-1, :]))
    q.communicate()
    f.close()
    return imgf


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
