# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import numpy as np
from raytraverse.lightfield import SunSkyPt
from raytraverse.integrator.integrator import Integrator


class SunSkyIntegrator(Integrator):
    """for merging SCBinField and SunField outputs, creates
    a SunSkyPt for integration at each point location during integrate()
    """

    def __init__(self, skyfield, sunfield, wea=None, loc=None, skyro=0.0,
                 ground_fac=0.15):
        #: raytraverse.sunsetter.SunSetter
        self.suns = sunfield.suns
        #: raytraverse.lightfield.SunField
        self.sunfield = sunfield
        super().__init__(skyfield, wea=wea, loc=loc, skyro=skyro,
                         ground_fac=ground_fac)

    def _prep_data(self, sunskyfield, skyv, sun, sunb, pi):
        """prepare arguments for hdr/metric computation"""
        if sunb[1] in sunskyfield.items() and sun[-2] > 0:
            lf = sunskyfield
            li = sunb[1]
            skyvec = np.concatenate((sun[-2:-1], skyv))
            return lf, li, skyvec
        else:
            return super()._prep_data(sunskyfield.skyparent, skyv, sun,
                                      sunb, pi)

    def pt_field(self, pi):
        return SunSkyPt(self.skyfield, self.sunfield, pi)
