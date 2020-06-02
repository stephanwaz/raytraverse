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

import numpy as np

from clasp import script_tools as cst
from clasp.click_callbacks import parse_file_list
from raytraverse import SpaceMapper, sunpos, translate


class Scene(object):
    """container for scene description

    Parameters
    ----------
    scene: str
        space separated list of radiance scene files (no sky) or octree
    area: str
        radiance scene file containing planar geometry of analysis area
    outdir: str
        path to store scene info and output files
    reload: bool, optional
        if True attempts to load existing scene files in new instance
        overrides 'overwrite'
    overwrite: bool, optional
        if True and outdir exists, will overwrite, else raises a FileExistsError
    wea: str, optional
        path to epw or wea file, if loc not set attempts to extract location
        data
    loc: (float, float, int), optional
        location data given as lat, lon, mer with + west of prime meridian
        overrides location data in wea
    ptro: float, optional
        angle in degrees counter-clockwise to point grid
    skyro: float, optional
        angle in degrees counter-clockwise to rotate sky
        (to correct model north, equivalent to clockwise rotation of scene)
    weaformat: {'time', 'angle'}, optional
        specify format of wea file:
            - 'time' - wea or epw file with or without header (requires loc)
              (default)
            - 'angle' - file format four number per line whitespace seperated
              (altitude, azimuth, direct normal radiation (W/m^2),
              diffuse horizontal radiation (W/m^2))
    """

    def __init__(self, scene, area, outdir, reload=False, overwrite=False,
                 wea=None, loc=None, ptro=0.0, skyro=0.0, weaformat='time'):
        try:
            os.mkdir(outdir)
        except FileExistsError as e:
            if reload:
                pass
            elif overwrite:
                shutil.rmtree(outdir)
                os.mkdir(outdir)
            else:
                raise e
        if weaformat.lower() not in ['time', 'angle']:
            raise ValueError("Invalid weaformat, choose from: 'time', 'angle'")
        #: {'time', 'angle'}: expected format for sky data
        self.weaformat = weaformat.lower()
        #: bool: try to reload scene files
        self.reload = reload
        #: float: ccw rotation (in degrees) for point grid on plane
        self.ptro = ptro
        #: float: ccw rotation (in degrees) for sky
        self.skyro = skyro
        #: str: path to store scene info and output files
        self.outdir = outdir
        self._solarbounds = None
        self.loc = loc
        self.skydata = wea
        self.scene = scene
        self.area = area
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
    def scene(self, scene):
        o = f'{self.outdir}/scene.oct'
        if self.reload and os.path.isfile(o):
            pass
        else:
            dims = cst.pipeline([f'getinfo -d {scene}', ])
            if re.match(scene + r': [\d.-]+ [\d.-]+ [\d.-]+ [\d.-]+',
                        dims.strip()):
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

    @property
    def skydata(self):
        """analysis area

        :getter: Returns this scene's skydata
        :setter: Sets this scene's skydata from file path
        :type: np.array
        """
        return self._skydata

    @skydata.setter
    def skydata(self, wea):
        sd = f'{self.outdir}/skydat.txt'
        if self.reload and os.path.isfile(sd):
            try:
                self._skydata = np.loadtxt(sd)
            except OSError:
                self._skydata = None
        elif wea is not None:
            if self.weaformat == 'time':
                if self.loc is None:
                    self.loc = sunpos.get_loc_epw(wea)
                wdat = sunpos.read_epw(wea)
                times = sunpos.row_2_datetime64(wdat[:,0:3])
                angs = sunpos.sunpos_degrees(times, *self.loc, ro=self.skyro)
                self._skydata = np.hstack((angs, wdat[:, 3:]))
            else:
                self._skydata = np.loadtxt(wea)
            np.savetxt(sd, self._skydata)
        else:
            self._skydata = None

    @property
    def solarbounds(self):
        """read only extent of solar bounds for given location
        set via loc

        :getter: Returns solar bounds
        :type: (np.array, np.array)
        """
        return self._solarbounds

    @property
    def loc(self):
        """scene location

        :getter: Returns location
        :setter: Sets location and self.solarbounds
        :type: (float, float, int)
        """
        return self._loc

    @loc.setter
    def loc(self, loc):
        """
        generate UV coordinates for jun 21 and dec 21 to use for masking
        sky positions
        """
        self._loc = loc
        if loc is not None:
            jun = np.arange('2020-06-21', '2020-06-22', 5,
                            dtype='datetime64[m]')
            dec = np.arange('2020-12-21', '2020-12-22', 5,
                            dtype='datetime64[m]')
            jxyz = sunpos.sunpos_xyz(jun, *loc, ro=self.skyro)
            dxyz = sunpos.sunpos_xyz(dec, *loc, ro=self.skyro)
            juv = translate.xyz2uv(jxyz[jxyz[:,2] > 0])
            duv = translate.xyz2uv(dxyz[dxyz[:,2] > 0])
            juv = juv[juv[:, 0].argsort()]
            duv = duv[duv[:, 0].argsort()]
            self._solarbounds = (juv, duv)
        else:
            self._solarbounds = None

    def in_solarbounds(self, uv):
        """
        default method for checking if src direction is in solar transit

        Parameters
        ----------
        uv: np.array
            source directions

        Returns
        -------
        result: np.array
            Truth of ray.src within solar transit
        """
        juv, duv = self.solarbounds
        vlow = duv[np.searchsorted(duv[:, 0], uv[:, 0]) - 1]
        vup = juv[np.searchsorted(juv[:, 0], uv[:, 0]) - 1]
        inbounds = np.logical_and(np.logical_and(vlow[:, 1] <= uv[:, 1],
                                  uv[:, 1] <= vup[:, 1]), uv[:, 0] < 1)
        return inbounds

    def in_area(self, uv):
        """check if point is in boundary path

        Parameters
        ----------
        xy: np.array
            world xy coordinates, shape (N, 2)

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
