# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import numpy as np

from raytraverse.integrator import IntegratorDS
from raytraverse.integrator.zonalintegrator import ZonalIntegrator


class ZonalIntegratorDS(IntegratorDS, ZonalIntegrator):
    """specialized integrator for 2-phase DDS style calculation. assumes
    first lightplane is sky contribution, second, direct sky contribution
    (with identical sampling to sky) and third direct sun contribution. Uses
    special point functions that combine two sky functions on a per patch basis.
    """
    def _match_ragged(self, smtx, dsns, sunidx, all_vecs):
        skyq = self.lightplanes[0].query(all_vecs[:, 3:])[0]
        tidxs = np.stack([skyq, skyq, sunidx])
        skydatas = [smtx, dsns[:, 4:], dsns[:, 3:4]]
        return tidxs, skydatas
