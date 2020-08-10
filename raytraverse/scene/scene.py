# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

import os
import re
import shutil
import configparser
import sys

import numpy as np
import json

from clasp import script_tools as cst
from clasp.click_callbacks import parse_file_list
from scipy.spatial import cKDTree

from raytraverse.mapper import SpaceMapper, ViewMapper


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
    reload: bool, optional
        if True attempts to load existing scene files in new instance
        overrides 'overwrite'
    overwrite: bool, optional
        if True and outdir exists, will overwrite, else raises a FileExistsError
    ptres: float, optional
        final spatial resolution in scene geometry units
    ptro: float, optional
        angle in degrees counter-clockwise to point grid
    viewdir: (float, float, float), optional
        vector (x,y,z) view direction (orients UV space)
    viewangle: float, optional
        should be 1-180 or 360
    skyres: float, optional
        approximate square patch size (sets sun resolution too)
    maxspec: float, optional
        maximum specular transmission in scene
        (used to clip pdf for sun sampling)
    """

    def __init__(self, outdir, scene=None, area=None, reload=True,
                 overwrite=False, ptres=1.0, ptro=0.0, viewdir=(0, 1, 0),
                 viewangle=360, skyres=10.0, maxspec=0.3, **kwargs):
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
            with open(js, 'w') as jf:
                json.dump(locvar, jf)
            #: bool: try to reload scene files
            self.reload = reload
            #: float: ccw rotation (in degrees) for point grid on plane
            self.ptro = ptro
            #: str: path to store scene info and output files
            self.outdir = outdir
            #: float: point resolution for area
            self.ptres = ptres
            #: float: maximum specular transmission in scene
            self.maxspec = maxspec
            self._solarbounds = None
            self.scene = scene
            self.area = area
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
                raise ValueError(f'{o} does not exist, Scene() must be invoked'
                                 ' with a scene= argument')
            if m:
                oconv = f'oconv -i {scene}'
            else:
                scene = " ".join(parse_file_list(None, scene))
                oconv = f'oconv -f {scene}'
            result, err = cst.pipeline([oconv, ],
                                       outfile=o,
                                       close=True, caperr=True, writemode='wb')
            if b'fatal' in err:
                raise ChildProcessError(err.decode(cst.encoding))
        self._scene = o

    @property
    def ptshape(self):
        """UV point resolution, set by area

        :getter: Returns this scenes's point resolution
        """
        return self._ptshape

    @property
    def area(self):
        """analysis area

        :getter: Returns this scenes's area
        :setter: Sets this scenes's area from file path
        :type: raytraverse.spacemapper.SpaceMapper
        """
        return self._area

    @area.setter
    def area(self, area):
        a = f'{self.outdir}/area.rad'
        if self.reload and os.path.isfile(a):
            pass
        else:
            shutil.copy(area, a)
        self._area = SpaceMapper(a, self.ptro)
        bbox = self.area.bbox[:, 0:2]/self.ptres
        size = (bbox[1] - bbox[0])
        self._ptshape = np.ceil(size).astype(int)

    @property
    def pt_kd(self):
        """point kdtree for spatial queries"""
        if self._pt_kd is None:
            self._pt_kd = cKDTree(self.pts())
        return self._pt_kd

    @pt_kd.setter
    def pt_kd(self, pt_kd):
        self._pt_kd = pt_kd

    def idx2pt(self, idx):
        shape = self.ptshape
        si = np.stack(np.unravel_index(idx, shape)).T
        return self.area.uv2pt((si + .5)/shape)

    def pts(self):
        shape = self.ptshape
        return self.idx2pt(np.arange(np.product(shape)))

    def in_area(self, uv):
        """check if point is in boundary path

        Parameters
        ----------
        uv: np.array
            uv coordinates, shape (N, 2)

        Returns
        -------
        mask: np.array
            boolean array, shape (N,)
        """
        path = self.area.path
        if path is None:
            return np.full((uv.shape[0]), True)
        else:
            result = np.empty((len(path), uv.shape[0]), bool)
            for i, p in enumerate(path):
                result[i] = p.contains_points(uv)
        return np.any(result, 0)

    def in_view(self, uv):
        """check if uv direction is in view

        Parameters
        ----------
        uv: np.array
            view uv coordinates, shape (N, 2)

        Returns
        -------
        mask: np.array
            boolean array, shape (N,)
        """
        inbounds = np.stack((uv[:, 0] >= 0, uv[:, 0] < self.view.aspect,
                             uv[:, 1] >= 0, uv[:, 1] < 1))
        return np.all(inbounds, 0)
