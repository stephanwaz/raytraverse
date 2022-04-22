# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import numpy as np
from raytraverse.sky.skydata import SkyData


class SkyDataMask(SkyData):
    """spoofed skydata class for use with light results

    Parameters
    ----------
    hours: np.array
        hours of year given as (m, d, h) where hour is H.5 (assumes 8760) to
        use as daymask.
    """

    def __init__(self, hours):
        #: sky patch resolution
        wea = []
        md = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        nh = 0
        for m in range(1, 13):
            days = md[m-1]
            for d in range(1, days+1):
                for h in range(24):
                    mdh = (m, d, h + 0.5)
                    if nh < len(hours) and np.allclose(hours[nh], mdh):
                        nh += 1
                        wea.append(mdh + (100, 100))
                    else:
                        wea.append(mdh + (0, 0))
        wea = np.asarray(wea)
        super().__init__(wea, loc=(0, 0, 0), skyro=0.0, ground_fac=0.2,
                         intersky=True, skyres=1, minalt=0.0, mindiff=50.0,
                         mindir=50.0)

    @property
    def skydata(self):
        """sun position and dirnorm diffhoriz"""
        return self._skydata

    @skydata.setter
    def skydata(self, wea):
        """spoof skydata

        Parameters
        ----------
        wea: np.array
            - 5 col: m, d, h, dir, diff
        """
        md = wea[:, 0:2].astype(int)
        self._rowlabel = wea[:, 0:3]
        self._daymask = wea[:, 4] > self._mindiff
        self._daysteps = np.sum(self.daymask)
        self._smtx = np.ones((self.daysteps, self.skyres**2 + 1))
        self._sun = np.ones((self.daysteps, 5))
        self._skydata = np.ones((8760, 5))
        self._sunproxy = np.zeros(self.daysteps)
