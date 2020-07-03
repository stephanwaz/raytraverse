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

