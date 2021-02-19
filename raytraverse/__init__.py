# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

"""Top-level package for raytraverse."""

__author__ = """Stephen Wasilewski"""
__email__ = 'stephanwaz@gmail.com'
__version__ = '1.1.2'
__all__ = ['crenderer', 'evaluate', 'formatter', 'lightpoint', 'mapper',
           'renderer', 'sampler', 'scene', 'sky', 'io', 'plot', 'translate']


def set_raypath(basefile=__file__, subd="cal"):
    import os
    if subd is not None:
        suff = os.path.sep + subd
    else:
        suff = ""
    raypath_rt = ['.', os.path.dirname(basefile) + suff]
    try:
        raypath_env = os.environ["RAYPATH"].split(os.pathsep)
    except KeyError:
        raypath_new = raypath_rt
    else:
        raypath_new = list(dict.fromkeys(raypath_rt + raypath_env))
    os.environ["RAYPATH"] = os.pathsep.join(raypath_new)


set_raypath()
