# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import os
import shutil


class Formatter(object):
    """scene formatter readies scene files for simulation, must be compatible
    with desired renderer.
    """
    #: line comment character
    comment = "#"

    #: arguments for direct trace
    direct_args = ""

    #: extension for renderer scene file
    scene_ext = ""

    @staticmethod
    def make_scene(scene_files, out, frozen=True):
        """compile scene"""
        if not os.path.isfile(scene_files):
            raise FileNotFoundError(f"{scene_files} must be an existing file")
        if frozen:
            try:
                shutil.copy(scene_files, out)
            except TypeError:
                raise ValueError('Cannot initialize Scene with '
                                 f'area={scene_files}')
        else:
            out = scene_files
        return out

    @staticmethod
    def add_source(scene, src):
        """add source files to compiled scene"""
        pass

    @staticmethod
    def get_skydef(color, ground=True, name='skyglow'):
        """assemble sky definition"""
        pass

    @staticmethod
    def get_sundef(vec, color, size=0.5333, mat_name='solar', mat_id='sun',
                   glow=False):
        """assemble sun definition"""
        pass

    @staticmethod
    def get_contribution_args(render_args, side, name):
        """prepare arguments for contribution based simulation"""
        pass

    @staticmethod
    def get_standard_args(render_args, ambfile=None):
        """prepare arguments for standard simulations"""
        pass

    @staticmethod
    def extract_sources(srcdef, accuracy):
        """scan scene file for sun source definitions"""
        return [], accuracy
