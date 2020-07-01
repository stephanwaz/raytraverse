# -*- coding: utf-8 -*-

# Copyright (c) 2019 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
from raytraverse.mapper.viewmapper import ViewMapper


class SunMapper(ViewMapper):
    """translate between view and normalized UV space

    Parameters
    ----------
    suns: np.array
        dx,dy,dz sun positions
    """

    def __init__(self, suns):
        super(SunMapper, self).__init__(suns, .533)
