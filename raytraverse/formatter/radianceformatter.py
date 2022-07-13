# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import re
import os

import numpy as np
from clasp import script_tools as cst
from clasp.click_callbacks import parse_file_list
from clasp.script_tools import pipeline

from raytraverse.formatter.formatter import Formatter


class RadianceFormatter(Formatter):
    """scene formatter readies scene files for simulation, must be compatible
    with desired renderer.
    """
    #: line comment character
    comment = "#"

    #: extension for renderer scene file
    scene_ext = ".oct"

    @staticmethod
    def make_scene(scene_files, out, frozen=True):
        """compile scene"""
        dims = cst.pipeline([f'getinfo -d {scene_files}', ])
        try:
            m = re.match(scene_files + r'.*: [\d.-]+ [\d.-]+ [\d.-]+ [\d.-]+',
                         dims.strip())
        except TypeError:
            raise ValueError(f'{scene_files} does not exist, Scene() must be '
                             'invoked with a scene= argument')
        if not frozen and m:
            out = scene_files
        else:
            if m:
                oconv = f'oconv -i {scene_files}'
            else:
                scene = " ".join(parse_file_list(None, scene_files))
                if frozen:
                    oconv = f'oconv -f {scene}'
                else:
                    oconv = f'oconv {scene}'
            result, err = cst.pipeline([oconv, ], outfile=out, close=True,
                                       caperr=True, writemode='wb')
            if b'fatal' in err:
                os.remove(out)
                raise ChildProcessError(err.decode(cst.encoding))
        return out

    @staticmethod
    def get_scene(scene):
        """recover scene file paths from compiled octree

        Parameters
        ----------
        scene: octree file

        Returns
        -------
        files: string to use in new octree generation. -i prepended before
        each actree
        frozen: if result will be a frozen octree
        """
        if not os.path.isfile(scene):
            raise FileNotFoundError(scene)
        hdr = pipeline([f"getinfo {scene}"])
        oconvf = re.findall(r"\s*oconv (.+)", hdr)[-1]
        files = oconvf.replace("-f ", "")
        frozen = "-i" in files
        filel = [i for i in files.split() if re.match(r".+\..+", i)]
        if not np.all([os.path.isfile(i) for i in filel]):
            files = f"-i {scene}"
        return files, frozen

    @staticmethod
    def get_skydef(color=(.96, 1.004, 1.118), ground=True, name='skyglow',
                   mod="void", groundname=None, groundcolor=(1, 1, 1)):
        """assemble sky definition"""
        if groundname is None:
            groundname = name
            groundmod = ""
        else:
            groundmod = (f"{mod} glow {groundname} 0 0 4 {groundcolor[0]} "
                         f"{groundcolor[1]} {groundcolor[2]} 0\n")
        skydeg = (f"{mod} glow {name} 0 0 4 {color[0]} {color[1]} {color[2]} 0"
                  f"\n{name} source sky 0 0 4 0 0 1 180\n")
        if ground:
            skydeg += f"{groundmod}{groundname} source ground  0 0 4 0 0 -1 180"
        return skydeg

    @staticmethod
    def get_sundef(vec, color, size=0.5333, mat_name='solar', mat_id='sun'):
        """assemble sun definition"""
        d = f"{vec[0]} {vec[1]} {vec[2]}"
        dec = (f"void light {mat_name} 0 0 3 {color[0]} {color[1]} {color[2]}\n"
               f"{mat_name} source {mat_id} 0 0 4 {d} {size}\n")
        return dec
