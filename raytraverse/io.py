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

import numpy as np


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
    lum = np.where(rgbe[:, 0:3] == 0, 0, (rgbe[:, 0:3] + 0.5) * v)
    # luminance = 179 * (0.265*R + 0.670*G + 0.065*B)
    return np.einsum('ij,j', lum, [47.435, 119.93, 11.635])
