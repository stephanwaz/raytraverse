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

    def __init__(self):
        # Create pipe and dup2() the write end of it on top of stdout,
        # saving a copy  of the old stdout
        self.fileno = sys.stdout.fileno()
        self.save = os.dup(self.fileno)
        self.pipe = os.pipe()
        os.dup2(self.pipe[1], self.fileno)
        os.close(self.pipe[1])
        self.stdout = ''
        self.threader = threading.Thread(target=self.drain_pipe)
        self.threader.start()

    def __enter__(self):
        return self

    def drain_pipe(self):
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
    r = craytraverse.Rtrace.getInstance()
    r.initialize(args)
    map(r.call, ['rays.txt']*10)
    r.call('rays.txt')
    r.call('rays.txt')
    print('capturing')
    with CaptureStdOut() as capture:
        r.call('rays.txt')
    print('Captured stdout:\n' + capture.stdout)

    time.sleep(1)
    # r = craytraverse.Rtrace.getInstance()
    with CaptureStdOut() as capture:
        r.call('rays.txt')
    print('Captured stdout:\n' + capture.stdout)

    time.sleep(1)
    # r = craytraverse.Rtrace.getInstance()
    with CaptureStdOut() as capture:
        r.call('rays.txt')
    print('Captured stdout:\n' + capture.stdout)
    r.reset()

    time.sleep(1)

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
