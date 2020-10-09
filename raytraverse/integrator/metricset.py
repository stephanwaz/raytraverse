# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import numpy as np


class MetricSet(object):

    def __init__(self, vm, *args, scale=179):
        self.vm = vm
        header = (" ".join(args)).split()
        afuncs = MetricSet.metricdict
        self.header = [h for h in header if h in afuncs]
        rfuncs = []
        keys = []
        for k, v in afuncs.items():
            if k in self.header:
                rfuncs.append(afuncs[k])
                keys.append(k)
        self.scale = scale
        self.funcs = tuple(rfuncs)
        self.sort = [keys.index(i) for i in self.header]
        self._c = {}

    # -------------------metric functions-------------------

    def illum(self, vec, omega, lum):
        ev = np.einsum('i,i,i->', self.vm.ctheta(vec), lum, omega)*self.scale
        self._c['Ev'] = ev
        return ev

    def avglum(self, vec, omega, lum):
        area = np.sum(omega)
        alum = np.einsum('i,i->', lum, omega)*self.scale/area
        self._c['La'] = alum
        self._c['O'] = area
        return alum

    def sqlum(self, vec, omega, lum):
        try:
            alum = self._c['La']
        except KeyError:
            alum = self.avglum(vec, omega, lum)
        a2lum = np.einsum('i,i,i->', lum, lum, omega)*self.scale**2/self._c['O']
        return a2lum/alum**2

    # ----------------end metric functions-------------------

    # must be populated in execution order
    metricdict = {"illum": illum, "avglum": avglum, "lum2": sqlum}

    def compute(self, vec, omega, lum):
        """

        Parameters
        ----------
        vec
        omega
        lum

        Returns
        -------

        """
        self._c.clear()
        r = [f(self, vec, omega, lum) for f in self.funcs]
        self._c.clear()
        return np.array(r)[self.sort]
