# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
from datetime import datetime, timezone
import os
import shutil
import sys

from raytraverse.formatter import Formatter
from raytraverse.utility import TStqdm


class BaseScene(object):
    """container for scene description

    Parameters
    ----------
    outdir: str
        path to store scene info and output files
    scene: str, optional (required if not reload)
        space separated list of radiance scene files (no sky) or octree
    frozen: bool, optional
        create a frozen octree
    formatter: raytraverse.formatter.Formatter, optional
        intended renderer format
    reload: bool, optional
        if True attempts to load existing scene files in new instance
        overrides 'overwrite'
    overwrite: bool, optional
        if True and outdir exists, will overwrite, else raises a FileExistsError
    log: bool, optional
        log progress events to outdir/log.txt
    loglevel: int, optional
        maximum sampler level to log
    """

    def __init__(self, outdir, scene=None, frozen=True, formatter=Formatter,
                 reload=True, overwrite=False, log=True, loglevel=10,
                 utc=False):
        self.outdir = outdir
        try:
            os.mkdir(outdir)
        except FileExistsError as e:
            if overwrite:
                shutil.rmtree(outdir)
                os.mkdir(outdir)
            elif reload:
                pass
            else:
                raise e
        except TypeError:
            log = False
        self._logf = f"{self.outdir}/log.txt"
        self._dolog = log
        if utc:
            self._tz = timezone.utc
        else:
            self._tz = None
        self._loglevel = loglevel
        self.formatter = formatter
        self._frozen = frozen
        self.reload = reload
        self.scene = scene
        self.reload = False

    @property
    def scene(self):
        """render scene files (octree)

        :getter: Returns this samplers's scene file path
        :setter: Sets this samplers's scene file path and creates run files
        :type: str
        """
        return self._scene

    @scene.setter
    def scene(self, scene_files):
        try:
            self._scene = f'{self.outdir}/scene{self.formatter.scene_ext}'
        except AttributeError:
            self._scene = None
        else:
            if self.outdir is None or (self.reload and
                                       os.path.isfile(self._scene)):
                pass
            elif scene_files:
                self._scene = self.formatter.make_scene(scene_files,
                                                        self._scene,
                                                        frozen=self._frozen)

    def log(self, instance, message, err=False, level=0):
        """print a message to the log file or stderr

        Parameters
        ----------
        instance: Any
            the parent class for the progress bar
        message: str, optional
            the message contents
        err: bool, optional
            print to stderr instead of self._logf
        level: int, optional
            the nested level of the message
        """
        if self._dolog and level <= self._loglevel:
            if level < 0:
                message = f"{type(instance).__name__}\t{message}"
            else:
                if level == 0:
                    tf = "%d-%b-%Y %H:%M:%S"
                else:
                    tf = "%H:%M:%S"
                ts = datetime.now(tz=self._tz).strftime(tf)
                indent = " | " * level
                message = f"{indent}{ts}\t{type(instance).__name__}\t{message}"
            f = sys.stderr
            needsclose = False
            if not err:
                try:
                    f = open(self._logf, 'a')
                    needsclose = True
                except TypeError:
                    pass
            if not needsclose:
                message = message.replace("\t", "  ")
            print(message, file=f)
            if needsclose:
                f.close()

    def progress_bar(self, instance, iterable=None, message=None,
                     total=None, level=0, workers=False):
        """generate a tqdm progress bar and concurrent.futures Executor class

        Parameters
        ----------
        instance: Any
            the parent class for the progress bar
        iterable: Sequence, optional
            passed to tqdm, the sequence to loop over
        message: str, optional
            the prefix message for the progress bar
        total: int, optional
            the number of expected iterations (when interable is none)
        level: int, optional
            the nested level of the progress bar
        workers: Union[bool, str], optional
            if "thread/threads/t" returns a ThreadPoolExecutor, else if True
            returns a ProcessPoolExecutor.


        Returns
        -------
        TStqdm:
            a subclass of tqdm that decorates messages and has a pool
            property for multiprocessing.

        Examples
        --------

        with an iterable::

            for i in self.scene.progress_bar(self, np.arange(10)):
                do stuff...

        with workers=True:

            with self.scene.progress_bar(self, total=len(jobs) workers=True) as pbar:
                exc = pbar.pool
                do stuff...
                pbar.update(1)

        """
        return TStqdm(instance, self._tz, workers=workers, iterable=iterable,
                      total=total, desc=message, leave=None, position=level)
