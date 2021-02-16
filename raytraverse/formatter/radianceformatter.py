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

from raytraverse import io
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
            m = re.match(scene_files + r': [\d.-]+ [\d.-]+ [\d.-]+ [\d.-]+',
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
                raise ChildProcessError(err.decode(cst.encoding))
        return out

    @staticmethod
    def add_source(scene, src):
        """add source files to compiled scene"""
        out = scene.rsplit(".", 1)[0] + "_sky.oct"
        if os.path.isfile(src):
            ocom = f'oconv -f -i {scene} {src}'
            inp = None
        else:
            ocom = f'oconv -f -i {scene} -'
            inp = src
        f = open(out, 'wb')
        cst.pipeline([ocom], outfile=f, inp=inp, close=True)
        return out

    @staticmethod
    def get_skydef(color, ground=True, name='skyglow'):
        """assemble sky definition"""
        skydeg = (f"void glow {name} 0 0 4 {color[0]} {color[1]} {color[2]}  0 "
                  f"{name} source sky 0 0 4 0 0 1 180\n")
        if ground:
            skydeg += f"{name} source ground  0 0 4 0 0 -1 180"
        return skydeg

    @staticmethod
    def get_sundef(vec, color, size=0.5333, mat_name='solar', mat_id='sun'):
        """assemble sun definition"""
        d = f"{vec[0]} {vec[1]} {vec[2]}"
        dec = (f"void light {mat_name} 0 0 3 {color[0]} {color[1]} {color[2]}\n"
               f"{mat_name} source {mat_id} 0 0 4 {d} {size}\n")
        return dec

    @staticmethod
    def extract_sources(srcdef, accuracy):
        """scan scene file for sun source definitions"""
        srcs = []
        srctxt = cst.pipeline([f"xform {srcdef}"])
        srclines = re.split(r"[\n\r]+", srctxt)
        for i, v in enumerate(srclines):
            if re.match(r"[\d\w]+\s+source\s+[\d\w]+", v):
                src = " ".join(srclines[i:]).split()
                srcd = np.array(src[6:10], dtype=float)
                if srcd[-1] < 3:
                    modsrc = " ".join(srclines[:i]).split()
                    modidx = next(j for j in reversed(range(len(modsrc)))
                                  if modsrc[j] == src[0])
                    modi = io.rgb2rad(np.array(modsrc[modidx + 4:modidx + 7],
                                               dtype=float))
                    srcs.append(np.concatenate((srcd, [modi])))
                    # 1/(np.square(0.2665 * np.pi / 180) * .5) = 92444
                    # the ratio of suns area to hemisphere
                    accuracy = accuracy*modi/92444
                    break
        return srcs, accuracy
