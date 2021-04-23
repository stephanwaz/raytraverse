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
                 reload=True, overwrite=False, log=True, loglevel=10, utc=True):
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
            else:
                self._scene = self.formatter.make_scene(scene_files,
                                                        self._scene,
                                                        frozen=self._frozen)

    def log(self, instance, message, err=False, level=0):
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
