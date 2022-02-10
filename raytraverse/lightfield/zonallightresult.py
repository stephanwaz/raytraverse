# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import numpy as np
from raytraverse import io
from scipy.spatial import cKDTree

from raytraverse.lightfield.lightresult import LightResult, ResultAxis
from raytraverse.mapper import PlanMapper


class RaggedResult(tuple):
    """has a shape parameter and indexing similar to a np.array, but with
    varying shape along the second axis. composed of a list of np.arrays whose
    shape match after the first dimension."""

    def __new__(cls, a):
        return tuple.__new__(RaggedResult, a)

    def __init__(self, a):
        self.itemshape = self[0][0].shape
        self.shape = (len(self), 1) + self.itemshape
        self.items = tuple([len(i) for i in self])

    def __getitem__(self, item):
        try:
            return super(RaggedResult, self).__getitem__(item)
        except TypeError:
            pass
        if not hasattr(item[0], 'shape') and item[0] == Ellipsis:
            item = tuple(slice(None) for i in range(1 + len(self.shape) - len(item))) + item[1:]
        try:
            x = super(RaggedResult, self).__getitem__(item[0])
        except TypeError:
            x = [super(RaggedResult, self).__getitem__(i) for i in item[0]]
        if type(item[0]) == int:
            return x[item[1:]]
        else:
            return RaggedResult([i[item[1:]] for i in x])


class ZonalLightResult(LightResult):
    """a semi-dense representation of lightfield data analyzed for a set of
    metrics

    this class handles writing and loading results to disk as binary data and
    intuitive result extraction and reshaping for downstream visualisation and
    analysis using one of the "pull" methods. axes are indexed both numerically
    and names for increased transparency and ease of use.

    """

    def __init__(self, data, *axes, pointmetrics=None):
        self.pointmetrics = pointmetrics
        if pointmetrics is None:
            self.pointmetrics = []
        super().__init__(data, *axes)
        namee = ["sky", "zone", "view", "metric"]
        if (len(namee) != len(self.names) or
                not np.all([i == j for i, j in zip(namee, self.names)])):
            raise ValueError(f"ZonalLightResult must be initialized with axes: "
                             f"{namee}, not {self.names}")

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, d):
        self._data = RaggedResult(d)

    def load(self, file):
        with np.load(file) as result:
            names = result['names']
            self.pointmetrics = result['pointmetrics']
            axes = tuple([ResultAxis(result[f"arr_{i}"], n)
                          for i, n in enumerate(names)])
            data = [result[f"data_{i}"] for i in range(len(axes[0].values))]
        return data, axes

    def write(self, file, compressed=True):
        kws = dict(names=self.names, pointmetrics=self.pointmetrics)
        for i in range(len(self.data)):
            kws[f"data_{i}"] = self.data[i]
        args = [a.values for a in self.axes]
        if compressed:
            np.savez_compressed(file, *args, **kws)
        else:
            np.savez(file, *args, **kws)
        if hasattr(file, "write"):
            self._file = file.name
            file.close()
        else:
            self._file = file

    def _pull_labels(self, data, order, preserve, filters):
        ax0_labels = []
        oshp = []
        for i in order[:-preserve]:
            if (self.names[i] == "sky" and
                    self._index("zone") in order[:-preserve]):
                filt = filters[self._index("sky")]
                slab = []
                for s, c in zip(self.axis("sky").values[filt],
                                np.asarray(self.data.items)[filt]):
                    slab += [s]*c
                slab = np.array(slab)
                ax0_labels.append(slab)
                oshp.append(range(len(slab)))
            elif self.names[i] != "zone":
                slab = self.axes[i].values[filters[i]]
                ax0_labels.append(slab)
                oshp.append(range(len(slab)))
        return oshp, ax0_labels

    def _transpose_and_shape(self, data, order, preserve):
        shp = [data.shape[i] for i in order[-preserve:]]
        if self._index("sky") in order[-preserve:]:
            o2 = [i - 1 for i in order if i != 0]
            s2 = [data.shape[i] for i in order[-preserve:] if i != 0]
            d2 = []
            for d in data:
                d2.append(np.transpose(d, o2).reshape(-1, *s2))
            result = RaggedResult(d2)
        else:
            data = np.concatenate(data)
            result = np.transpose(data, [i if i == 0 else i - 1 for i in order
                                         if i != 1]).reshape(-1, *shp)
        return result

    def _pad_order(self, axes, preserve, **kwargs):
        # get numeric indices of keeper axes
        order = super()._pad_order(axes, preserve)
        # self._check_pull_params
        if self._index("zone") in order[-preserve:]:
            raise ValueError("Cannot pull along ragged axis 'zone'")
        if 'zone' in kwargs:
            raise ValueError("Cannot filter ragged axis 'zone'")
        if order[-preserve] == self._index("sky"):
            raise ValueError("sky cannot be output column with ragged points")
        return order

    def _print_serial(self, rt, labels, names, basename, header,
                      rowlabel, skyfill):
        if type(rt) == RaggedResult:
            flabels = self.fmt_names(names[-1], labels[-1])
            rowlabels = self.row_labels(labels[0])
            for i, j in enumerate(flabels):
                if names[0][0:4] == "zone":
                    rls = np.tile(rowlabels, int(rt[i].shape[0]/len(rowlabels)))
                else:
                    rls = np.repeat(rowlabels,
                                    int(rt[i].shape[0]/len(rowlabels)))
                f = open(f"{basename}_{j}.txt", 'w')
                self._print(f, rt[i], header, rls, rowlabel)
                f.close()
        else:
            super()._print_serial(rt, labels, names, basename, header,
                                  rowlabel, None)

    def pull2hdr(self, imgzone, basename, **kwargs):
        pm = PlanMapper(imgzone)
        img, vecs, mask, mask2, header = pm.init_img(480)
        print(img.shape, vecs[mask].shape, img[mask].shape)
        if "metric" in kwargs and kwargs["metric"] is not None:
            kwargs["metric"] = np.unique(np.concatenate(([0, 1, 2], kwargs["metric"])))
        rt, labels, names = self.pull("metric", preserve=2, **kwargs)
        flabels0 = self.fmt_names(names[-1], labels[-1])
        flabels1 = self.fmt_names(names[-2], labels[-2][3:])
        print(flabels1)
        for i, la0 in enumerate(flabels0):
            data = rt[i]
            kd = cKDTree(data[:, 0:3])
            data = data[:, 3:]
            err, idx = kd.query(vecs[mask])
            for k, (la, d) in enumerate(zip(flabels1, data.T)):
                img[mask] = d[idx]
                io.array2hdr(img, f"{basename}_{la}_{la0}.hdr", header)
