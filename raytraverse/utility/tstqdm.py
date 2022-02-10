# -*- coding: utf-8 -*-

# Copyright (c) 2019 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

"""progress bar"""
import shutil
from datetime import datetime

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures import wait, FIRST_COMPLETED

from tqdm import tqdm
from raytraverse import io


class TStqdm(tqdm):

    def __init__(self, instance=None, tz=None, workers=False, position=0,
                 desc=None, ncols=100, cap=None, **kwargs):
        if str(workers).lower() in ('thread', 't', 'threads'):
            pool = ThreadPoolExecutor()
        elif workers:
            nproc = io.get_nproc(cap)
            pool = ProcessPoolExecutor(nproc)
        else:
            pool = None
        self._instance = instance
        self.loglevel = position
        tf = "%H:%M:%S"
        self.ts = datetime.now(tz=tz).strftime(tf)
        self.pool = pool
        self.wait = wait
        self.FIRST_COMPLETED = FIRST_COMPLETED
        if pool is None:
            self.nworkers = 0
        else:
            self.nworkers = pool._max_workers
        ncols = min(ncols, shutil.get_terminal_size().columns)
        super().__init__(desc=self.ts_message(desc), position=position,
                         ncols=ncols, **kwargs)

    def ts_message(self, s):
        if self._instance is not None:
            p = type(self._instance).__name__
        else:
            p = ""
        if s is None:
            s = f"{p}"
        else:
            s = f"{p} {s}"
        s = f"{' | ' * self.loglevel} {s}"
        return s

    def write(self, s, file=None, end="\n", nolock=False):
        super().write(self.ts_message(s), file, end, nolock)

    def set_description(self, desc=None, refresh=True):
        super().set_description(desc=self.ts_message(desc), refresh=refresh)



