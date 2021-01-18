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
    """

    def __init__(self, outdir, scene=None, frozen=True, formatter=Formatter,
                 reload=True, overwrite=False, log=True, **kwargs):
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
        try:
            os.mkdir(outdir)
        except FileExistsError as e:
            pass
        self._logf = f"{self.outdir}/log.txt"
        self._dolog = log
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
        o = f'{self.outdir}/scene{self.formatter.scene_ext}'
        if self.reload and os.path.isfile(o):
            pass
        else:
            o = self.formatter.make_scene(scene_files, o, frozen=self._frozen)
        self._scene = o

    def log(self, instance, message, err=False):
        if self._dolog:
            f = sys.stderr
            needsclose = False
            if not err:
                try:
                    f = open(self._logf, 'a')
                    needsclose = True
                except TypeError:
                    pass
            ts = datetime.now(tz=timezone.utc).strftime("%d-%b-%Y %H:%M:%S")
            print(f"{ts}\t{type(instance).__name__}\t{message}", file=f)
            if needsclose:
                f.close()
