# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
from datetime import datetime, timezone
import os
import re
import shutil
import sys

import numpy as np
import json

from clasp import script_tools as cst
from clasp.click_callbacks import parse_file_list

from raytraverse.mapper import SpaceMapper, ViewMapper, SpaceMapperPt


class Scene(object):
    """container for scene description

    Parameters
    ----------
    outdir: str
        path to store scene info and output files
    scene: str, optional (required if not reload)
        space separated list of radiance scene files (no sky) or octree
    area: str, optional (required if not reload)
        radiance scene file containing planar geometry of analysis area
        or a list of points (line per point, space seperated, first 3 columns
        x, y, z
    reload: bool, optional
        if True attempts to load existing scene files in new instance
        overrides 'overwrite'
    overwrite: bool, optional
        if True and outdir exists, will overwrite, else raises a FileExistsError
    ptres: float, optional
        final spatial resolution in scene geometry units
    ptro: float, optional
        angle in degrees counter-clockwise to point grid
    pttol: float, optional
        tolerance for point search when using point list for area
    viewdir: (float, float, float), optional
        vector (x,y,z) view direction (orients UV space)
    viewangle: float, optional
        should be 1-180 or 360
    skyres: float, optional
        approximate square patch size in degrees
    maxspec: float, optional
        maximum specular transmission in scene
        (used to clip pdf for sun sampling)
    frozen: bool, optional
        create a frozen octree
    """

    def __init__(self, outdir, scene=None, area=None, reload=True,
                 overwrite=False, ptres=1.0, ptro=0.0, pttol=1.0,
                 viewdir=(0, 1, 0), viewangle=360, skyres=10.0, maxspec=0.3,
                 frozen=True, **kwargs):
        locvar = locals()
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
        js = f'{outdir}/scene_parameters.json'
        if os.path.isfile(js):
            with open(js, 'r') as jf:
                params = json.load(jf)
            os.remove(js)
            print(f'Scene parameters loaded from {js}', file=sys.stderr)
            self.__init__(**params)
        else:
            locvar.pop('self')
            locvar.pop('kwargs')
            #: bool: try to reload scene files
            self.reload = reload
            #: str: path to store scene info and output files
            self.outdir = outdir
            self._logf = f"{self.outdir}/log.txt"
            print(f"logging to {self._logf}", file=sys.stderr)
            self.log(self, f"Initializing {outdir}")
            a = f'{self.outdir}/area.rad'
            if self.reload and os.path.isfile(a):
                pass
            else:
                try:
                    shutil.copy(area, a)
                except TypeError:
                    raise ValueError('Cannot initialize Scene with '
                                     f'area={area}')
            try:
                ptload = np.loadtxt(a)[:, 0:3]
            except IndexError:
                ptload = np.loadtxt(a)[0:3].reshape(1, 3)
                self.area = SpaceMapperPt(a, ptres, ptro, pttol)
            except ValueError:
                self.area = SpaceMapper(a, ptres, ptro, pttol)
            else:
                self.area = SpaceMapperPt(a, ptres, ptro, pttol)
            #: float: maximum specular transmission in scene
            self.maxspec = maxspec
            self._solarbounds = None
            self._frozen = frozen
            self.scene = scene
            self.reload = False
            #: raytraverse.viewmapper.ViewMapper: view translation class
            self.view = ViewMapper(viewdir, viewangle)
            if skyres < .7:
                print('Warning! minimum sunres is .7 to avoid overlap and',
                      file=sys.stderr)
                print('allow for jittering position, sunres set to .7',
                      file=sys.stderr)
                skyres = .7
            self.skyres = skyres
            self.pt_kd = None
            locvar['scene'] = self.scene
            with open(js, 'w') as jf:
                json.dump(locvar, jf)

    def __del__(self):
        try:
            self.log(self, "Closed")
        except FileNotFoundError:
            pass

    @property
    def skyres(self):
        return self._skyres

    @skyres.setter
    def skyres(self, s):
        self._skyres = int(np.floor(90/s)*2)

    @property
    def scene(self):
        """render scene files (octree)

        :getter: Returns this samplers's scene file path
        :setter: Sets this samplers's scene file path and creates run files
        :type: str
        """
        return self._scene

    @scene.setter
    def scene(self, scene):
        o = f'{self.outdir}/scene.oct'
        if self.reload and os.path.isfile(o):
            pass
        else:
            dims = cst.pipeline([f'getinfo -d {scene}', ])
            try:
                m = re.match(scene + r': [\d.-]+ [\d.-]+ [\d.-]+ [\d.-]+',
                             dims.strip())
            except TypeError:
                raise ValueError(f'{scene} does not exist, Scene() must be '
                                 'invoked with a scene= argument')
            if not self._frozen and m:
                o = scene
            else:
                if m:
                    oconv = f'oconv -i {scene}'
                else:
                    scene = " ".join(parse_file_list(None, scene))
                    if self._frozen:
                        oconv = f'oconv -f {scene}'
                    else:
                        oconv = f'oconv {scene}'
                result, err = cst.pipeline([oconv, ], outfile=o, close=True,
                                           caperr=True, writemode='wb')
                if b'fatal' in err:
                    raise ChildProcessError(err.decode(cst.encoding))
        self._scene = o

    def pts(self):
        return self.area.pts()

    def log(self, instance, message):
        f = open(self._logf, 'a')
        ts = datetime.now(tz=timezone.utc).strftime("%d-%b-%Y %H:%M:%S")
        print(f"{ts}\t{type(instance).__name__}\t{message}", file=f)
        f.close()
