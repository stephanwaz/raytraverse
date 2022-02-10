# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import functools


from raytraverse.lightpoint import LightPointKD


class LightSet(object):

    def __init__(self, dataclass, scene, points, idx, **kwargs):
        self.scene = scene
        self.points = points
        self.idx = idx
        self.kwargs = kwargs
        self.dataclass = dataclass

    @functools.lru_cache(5)
    def __getitem__(self, item):
        return self.dataclass(self.scene, **self.kwargs)

    def __len__(self):
        return len(self.idx)


class LightPointSet(LightSet):
    """a collection of LightPoints, initialized by getitem"""

    def __init__(self, scene, points, idx, src, parent):
        super().__init__(LightPointKD, scene, points, idx, src=src,
                         parent=parent)

    @functools.lru_cache(5)
    def __getitem__(self, item):
        return self.dataclass(self.scene, pt=self.points[item],
                              posidx=self.idx[item], **self.kwargs)


class MultiLightPointSet(LightSet):
    def __init__(self, scene, points, idx, src, parent):
        super().__init__(LightPointKD, scene, points, idx, src=src,
                         parent=parent)
        self.src = self.kwargs.pop("src")

    @functools.lru_cache(5)
    def __getitem__(self, item):
        source = f"{self.src}_{self.idx[item, 0]:04d}"
        return self.dataclass(self.scene, pt=self.points[item, 3:],
                              posidx=self.idx[item, 1], src=source,
                              srcdir=self.points[item, 0:3], **self.kwargs)

