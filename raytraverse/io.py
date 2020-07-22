# -*- coding: utf-8 -*-

# Copyright (c) 2019 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

"""functions for reading and writing"""
import shlex
from subprocess import Popen, PIPE

import numpy as np

from raytraverse import optic


def call_sampler(outf, command, vecs):
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

    Returns
    -------
    lums: np.array
        of length vectors.shape[0]

    """
    f = open(outf, 'a+b')
    lum_file_pos = f.tell()
    p = Popen(shlex.split(command), stdout=f, stdin=PIPE)
    p.communicate(np2bytes(vecs))
    f.seek(lum_file_pos)
    lum = optic.rgb2rad(bytes2np(f.read(), (-1, 3)))
    f.close()
    return lum


def call_generic(commands, n=1):
    pops = []
    stdin = None
    for c in commands:
        pops.append(Popen(shlex.split(c), stdin=stdin, stdout=PIPE))
        stdin = pops[-1].stdout
    a = stdin.read()
    return np.fromstring(a, sep=' ').reshape(-1, n)


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
    """format ar as bytestring

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
    pval = f'pvalue -r -b -h -H -df -o -y {ar.shape[0]} +x {ar.shape[1]}'
    if header is not None:
        hdr = "' '".join(header)
        getinfo = shlex.split(f"getinfo -a '{hdr}'")
        p = Popen(pval.split(), stdin=PIPE, stdout=PIPE)
        q = Popen(getinfo, stdin=p.stdout, stdout=f)
    else:
        p = Popen(pval.split(), stdin=PIPE, stdout=f)
        q = p
    p.stdin.write(np2bytes(ar[-1::-1, -1::-1]))
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
