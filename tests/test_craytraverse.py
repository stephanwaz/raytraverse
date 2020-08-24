#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for raytraverse.craytraverse"""

from raytraverse import craytraverse
import numpy as np
from memory_profiler import profile

import os
import sys
import threading
from concurrent.futures import ProcessPoolExecutor
import time


class CaptureStdOut:

    def __init__(self, b=False):
        # Create pipe and dup2() the write end of it on top of stdout,
        # saving a copy  of the old stdout
        self.bytes = b
        self.fileno = sys.stdout.fileno()
        self.save = os.dup(self.fileno)
        self.pipe = os.pipe()
        os.dup2(self.pipe[1], self.fileno)
        os.close(self.pipe[1])
        if b:
            self.stdout = b''
            self.threader = threading.Thread(target=self.drain_bytes)
        else:
            self.stdout = ''
            self.threader = threading.Thread(target=self.drain_str)
        self.threader.start()

    def __enter__(self):
        return self

    def drain_bytes(self):
        while True:
            data = os.read(self.pipe[0], 1024)
            if not data:
                break
            self.stdout += data

    def drain_str(self):
        while True:
            data = os.read(self.pipe[0], 1024)
            if not data:
                break
            self.stdout += data.decode()

    def __exit__(self, exc_type, exc_value, traceback):
        # Close the write end of the pipe to unblock the reader thread and
        # trigger it to exit
        os.close(self.fileno)
        self.threader.join()
        # Clean up the pipe and restore the original stdout
        os.close(self.pipe[0])
        os.dup2(self.save, self.fileno)
        os.close(self.save)


@profile
def main():
    args = "rtrace -n 4 -ab 3 -ar 600 -ad 2000 -aa .2 -as 1500 -I test_run/sky.oct".split()
    print('Start')
    r = craytraverse.Rtrace.get_instance()
    r.initialize(args)
    r.call('rays.txt')
    r.update_ospec('Z', 'd')
    with CaptureStdOut(True) as capture:
        r.call('rays.txt')
    print(b'Captured stdout:\n' + capture.stdout)
    r.update_ospec('ZL', 'a')
    print('capturing')
    with CaptureStdOut() as capture:
        r.call('rays.txt')
    print('Captured stdout:\n' + capture.stdout)

    r.reset()
    r.initialize(args)
    r = craytraverse.Rtrace.get_instance()
    with CaptureStdOut() as capture:
        r.call('rays.txt')
    print('Captured stdout:\n' + capture.stdout)

    # r = craytraverse.Rtrace.getInstance()
    with CaptureStdOut() as capture:
        r.call('rays.txt')
    print('Captured stdout:\n' + capture.stdout)
    r.reset()
    r.initialize(args)

    # r = craytraverse.Rtrace.getInstance()
    with CaptureStdOut() as capture:
        r.call('rays.txt')
    print('Captured stdout:\n' + capture.stdout)
    r.reset()
    #
    # with CaptureStdOut() as capture:
    #     r.call('rays.txt')
    # print('Captured stdout:\n' + capture.stdout)
    #
    # print('done')




main()
