# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

"""c++ renderer objects"""

__all__ = ['cRcontrib', 'cRtrace']

import sys
import subprocess


def _find_radiance(executable):
    try:
        versiontxt = subprocess.Popen(f"{executable} -version".split(),
                                      stdout=subprocess.PIPE).communicate()[0]
    except FileNotFoundError:
        version = None
    else:
        version = versiontxt.split()[1][0]
    if version != '5':
        print(f'Warning: {executable} from radiance 5 not installed or env is not'
              ' properly set, SPRcontrib will not run')


try:
    from raytraverse.crenderer.rcontrib_c import cRcontrib
except (ModuleNotFoundError, ImportError):
    from raytraverse.renderer.sprenderer import SPRcontrib as cRcontrib
    print("Warning: No cRenderer found, falling back to SPRenderer for"
          " rcontrib", file=sys.stderr)
    _find_radiance("rcontrib")


try:
    from raytraverse.crenderer.rtrace_c import cRtrace
except (ModuleNotFoundError, ImportError):
    from raytraverse.renderer.sprenderer import SPRtrace as cRtrace
    print("Warning: No cRenderer found, falling back to SPRenderer for"
          " rcontrib", file=sys.stderr)
    _find_radiance("rtrace")

