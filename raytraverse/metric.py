# -*- coding: utf-8 -*-

# Copyright (c) 2019 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

"""standardized metric functions"""
import numpy as np


def illum(vm, vec, omega, lum, scale=179, **kwargs):
    return np.einsum('i,i,i->', vm.ctheta(vec), lum, omega) * scale


def avglum(vm, vec, omega, lum, scale=179, area=None, **kwargs):
    if area is None:
        area = np.sum(omega)
    return np.einsum('i,i->', lum, omega) * scale / area


def sqlum(vm, vec, omega, lum, scale=179, area=None, **kwargs):
    if area is None:
        area = np.sum(omega)
    alum2 = avglum(vm, vec, omega, lum, scale=scale, area=area, **kwargs)**2
    a2lum = np.einsum('i,i,i->', lum, lum, omega) * scale**2 / area
    return a2lum / alum2
