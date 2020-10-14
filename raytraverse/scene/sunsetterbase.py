# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import os
import numpy as np
from raytraverse.mapper import ViewMapper


class SunSetterBase(object):
    """bare bones class for on the fly sunsetter.

    Parameters
    ----------
    scene: raytraverse.scene.Scene
        scene class containing geometry, location and analysis plane
    suns: np.array
        sun (N, 5) positions, sizes, and intensities
    """

    def __init__(self, scene, suns=None, prefix="static_sources"):
        sunfile = f"{scene.outdir}/{prefix}.rad"
        self.scene = scene
        if suns is not None:
            self.suns = suns[:, 0:3]
            self.srct = np.min(suns[:, 4])/2
            self.srcsize = np.max(suns[:, 3])
            self._write_suns(sunfile)
        elif os.path.isfile(sunfile):
            f = open(sunfile, 'r')
            header = f.readline().strip()
            self.srct = float(header.split("=")[-1])
            sund = f.read().split('source')[1:]
            xyz = np.array([s.split()[4:8] for s in sund]).astype(float)
            self.suns = xyz[:, 0:3]
            self.srcsize = np.max(xyz[:, 3])
        else:
            raise ValueError("Cannot Initialize SunSetterBase without existing"
                             f"sun file ({sunfile}) or suns argument")
        self.map = ViewMapper(self.suns, self.srcsize, name=prefix)

    def write_sun(self, i):
        s = self.suns[i]
        mod = f"solar{i:05d}"
        name = f"sun{i:05d}"
        d = f"{s[0]} {s[1]} {s[2]}"
        dec = f"void light {mod} 0 0 3 1 1 1\n"
        dec += f"{mod} source {name} 0 0 4 {d} {self.srcsize}\n"
        return dec

    def _write_suns(self, sunfile):
        """write suns to file

        Parameters
        ----------
        sunfile
        """
        if self.suns.size > 0:
            f = open(sunfile, 'w')
            print(f"# srct={self.srct}", file=f)
            for i in range(self.suns.shape[0]):
                dec = self.write_sun(i)
                print(dec, file=f)
            f.close()
