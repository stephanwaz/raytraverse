# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import os.path
import re

import numpy as np
from scipy.spatial import cKDTree

from raytraverse.mapper import PlanMapper
from raytraverse import io


class ResultAxis(object):

    def __init__(self, values, name, cols=None):
        self.values = np.asarray(values)
        if len(self.values.shape) == 2 and self.values.shape[-1] == 1:
            self.values = self.values.ravel()
        if len(self.values.shape) == 2:
            dt = type(self.values.flat[0])
            dtype = np.dtype([(f"f{i}", dt) for i in
                              range(self.values.shape[1])])
            self.values = np.array(list(zip(*self.values.T)), dtype=dtype)
        self.name = name
        self.cols = cols

    def value_array(self):
        if len(self.values.dtype) > 0:
            return np.asarray([tuple(i) for i in self.values])
        else:
            return np.asarray([(i, ) for i in self.values])

    def index(self, i):
        idx = np.squeeze(np.where(self.values == i))
        if idx.size == 0:
            raise IndexError(f"'{i}' not in ResultAxis '{self.name}'")
        return idx

    @property
    def cols(self):
        return self._cols

    @cols.setter
    def cols(self, c):
        pt = ('x', 'y', 'z', 'dx', 'dy', 'dz')
        col_names = dict(point={2: pt[0:2], 3:pt[0:3],
                                4: pt[0:3] + ("area",), 6: pt},
                         view={3: pt[3:]},
                         sky={3: ('month', 'day', 'hour')})
        ex = self.values[0]
        try:
            if hasattr(ex, "upper"):
                cnt = 1
            else:
                cnt = len(ex)
        except TypeError:
            cnt = 1
        if c is None:
            if cnt == 1:
                c = (self.name,)
            else:
                try:
                    c = col_names[self.name][cnt]
                except KeyError:
                    c = list(range(cnt))
        elif not hasattr(c, "__len__") or hasattr(c, "upper"):
            c = (c,)
        if len(c) != cnt:
            raise ValueError(f"cols {c} does not match shape of values {cnt}")
        self._cols = "\t".join([str(i) for i in c])


class LightResult(object):
    """a dense representation of lightfield data analyzed for a set of metrics

    this class handles writing and loading results to disk as binary data and
    intuitive result extraction and reshaping for downstream visualisation and
    analysis using one of the "pull" methods. axes are indexed both numerically
    and names for increased transparency and ease of use.

    Parameters
    ----------
    data: np.array str
        multidimensional array of result data or file path to saved LightResule
    axes: Sequence[raytraverse.lightfield.ResultAxis]
        axis information
    """

    def __init__(self, data, *axes):
        self._file = None
        if not hasattr(data, "shape") and type(data) not in (list, tuple):
            if os.path.isfile(data):
                self._file = data
                data, axes = self.load(data)
            else:
                raise FileNotFoundError(f"{data} is not a valid file path")
        self.data = data
        self._axes = axes
        if len(self.data.shape) != len(self.axes):
            raise ValueError(f"data of shape: {self.data.shape} requires "
                             f"{len(self.data.shape)} axes arguments.")
        self._names = [a.name for a in axes]

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, d):
        self._data = d

    @property
    def axes(self):
        return self._axes

    @property
    def names(self):
        return self._names

    @property
    def file(self):
        return self._file

    def axis(self, name):
        return self.axes[self._index(name)]

    def load(self, file):
        with np.load(file) as result:
            data = result['data']
            names = result['names']
            axes = tuple([ResultAxis(result[f"arr_{i}"], n)
                          for i, n in enumerate(names)])
        return data, axes

    def write(self, file, compressed=True):
        kws = dict(data=self.data, names=self.names)
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

    def _merge_axes(self, *lrs, axis="sky"):
        if axis not in self.names:
            raise ValueError(f"axis {axis} is not in {type(self)}")
        outaxes = []
        filters = []
        for name in self.names:
            if name == axis:
                values = self.axis(axis).values
                for lr in lrs:
                    ov = lr.axis(axis).values
                    filt = np.logical_not(np.isin(ov, values))
                    values = np.concatenate([values, ov[filt]])
                    filters.append(filt)
                outaxes.append(ResultAxis(values, self.axis(axis).name))
            else:
                outaxes.append(self.axis(name))
                ca = self.axis(name).values
                for other in lrs:
                    ct = other.axis(name).values
                    mismatch = ValueError(f"axis {name} of {other} "
                                          f"does not match {self}")
                    if ca.shape != ct.shape:
                        raise mismatch
                    try:
                        match = np.allclose(ca, ct)
                    except TypeError:
                        match = np.all([a == b for a, b in zip(ca, ct)])
                    if not match:
                        raise mismatch
        return outaxes, filters

    def merge(self, *lrs, axis="sky"):
        """create merged lightresult from lightresults, must match on all axes
         except axis. does not sort but culls duplicates"""
        outaxes, filters = self._merge_axes(*lrs, axis=axis)
        oi = self._index(axis)
        data = self.data
        for lr, f, in zip(lrs, filters):
            if np.all(f):
                od = lr.data
            else:
                od = np.compress(f, lr.data, axis=oi)
            data = np.concatenate([data, od], axis=oi)
        return LightResult(data, *outaxes)

    def pull(self, *axes, preserve=1, **kwargs):
        """arrange and extract data slices from result.

        Integrators construct a light result with these axes:

            0. sky
            1. point
            2. view
            3. metric

        Parameters
        ----------
        axes: Union[int, str]
            the axes (by name or integer index) to reorder output, list will
            fill with default object order.
        preserve: int, optional
            number of dimensions to preserve (result will be N+1).
        kwargs:
            keys with axis names will be used to filter output.

        Returns
        -------
        result: np.array
            the result array, will have 1+len(axes) dims, with the shaped
            determined by axis size and any indices argument.
        labels: Sequence
            list of labels for each axis, for flattened axes will be a tuple
            of broadcast axis labels.
        names: Sequence
            list of strings of returned axis names
        """
        order = self._pad_order(axes, preserve, **kwargs)
        data, filters = self._filter_data(**kwargs)
        result = self._transpose_and_shape(data, order, preserve)

        # get axes names
        ax0_name = "_".join([self.names[i] for i in order[:-preserve]])
        names = [ax0_name] + [self.names[i] for i in order[-preserve:]]

        # get orginal shape of flattten axes to unravel labels
        oshp, ax0_labels = self._pull_labels(data, order, preserve, filters)
        ij = np.meshgrid(*oshp, indexing='ij')
        ax0_labels = [slab[i].ravel() for slab, i in zip(ax0_labels, ij)]
        labels = ([list(zip(*ax0_labels))] +
                  [self.axes[i].values[filters[i]] for i in order[-preserve:]])
        return result, labels, names

    def _pull_labels(self, data, order, preserve, filters):
        oshp = [range(data.shape[i]) for i in order[:-preserve]]
        ax0_labels = [self.axes[i].values[filters[i]] for i in
                      order[:-preserve]]
        return oshp, ax0_labels

    def _transpose_and_shape(self, data, order, preserve):
        shp = [data.shape[i] for i in order[-preserve:]]
        return data.transpose(order).reshape(-1, *shp)

    def _pad_order(self, axes, preserve, **kwargs):
        # get numeric indices of keeper axes
        order = [self._index(i) for i in axes]
        # fill out rest of order with default
        order += [i for i in range(len(self.names)) if i not in order]
        # flipped preserved axes to end
        order = order[preserve:] + order[:preserve]
        return order

    def _filter_data(self, **kwargs):
        filters = []
        ns = slice(None)
        data = self.data
        for i, n in enumerate(self.names):
            filt = [ns]*len(self.data.shape)
            if n in kwargs and kwargs[n] is not None:
                if type(kwargs[n]) == int:
                    filt[i] = slice(kwargs[n], kwargs[n] + 1)
                elif type(kwargs[n]) == slice or len(kwargs[n]) > 1:
                    filt[i] = kwargs[n]
                elif len(kwargs[n]) == 1:
                    filt[i] = slice(kwargs[n][0], kwargs[n][0] + 1)
            data = data[tuple(filt)]
            filters.append(filt[i])
        return data, filters

    @staticmethod
    def _print(file, rt, header, rls, rowlabel=True):
        if header:
            print(header, file=file)
        for r, rh in zip(rt, rls):
            rl = "\t".join([f"{i:.05g}" for i in r])
            if rowlabel:
                rl = rh + "\t" + rl
            print(rl, file=file)

    @staticmethod
    def row_labels(labels):
        rls = []
        for rh in labels:
            rl = "\t".join(str(i) for i in rh)
            rl = re.sub(r"[(){}\[\]]", "", rl).replace(", ", "\t")
            rls.append(rl)
        return rls

    @staticmethod
    def fmt_names(name, labels):
        rls = []
        try:
            tl = len(labels[0].dtype) > 0
        except TypeError:
            tl = False
        for rh in labels:
            if tl:
                rl = "_".join(f"{i:05.02f}" for i in rh)
            elif np.issubdtype(type(rh), np.integer):
                rl = f"{rh:04d}"
            else:
                rl = rh
            rls.append(f"{name}_{rl}")
        return rls

    def pull_header(self, names, labels, rowlabel=True):
        h = [str(i) for i in labels[1]]
        if rowlabel:
            h = [self.axis(i).cols for i in names[0].split("_")] + h
        return "\t".join(h)

    def print(self, col, header=True, rowlabel=True, file=None,
              skyfill=None, **kwargs):
        """first calls pull and then prints 2d result to file"""
        rt, labels, names = self.pull(*col, **kwargs)
        if header:
            header = self.pull_header(names, labels, rowlabel)
        rowlabels = self.row_labels(labels[0])
        if skyfill is not None:
            self._check_sky_data(col[0], skyfill, rowlabels)
            rt = skyfill.fill_data(rt)
            rowlabels = self.row_labels(skyfill.rowlabel)
            if header:
                header = self.pull_header(["sky"], labels, rowlabel)
        self._print(file, rt, header, rowlabels, rowlabel)

    def sky_percentile(self, metric, per=(50,), **kwargs):
        mi = np.flatnonzero([i == metric for i in self.axis("metric").values])
        rt, labels, names = self.pull(self._index("metric"), self._index("sky"), preserve=2, metric=mi,
                                      **kwargs)
        rt = np.squeeze(np.percentile(rt, per, -1), -1).T
        labels = [[tuple(i)+tuple(j) for i, j in labels[0]], [f"{metric}_{p}"for p in per]]
        names = [names[0], names[-2]]

        axes = [ResultAxis(la, na) for la, na in zip(labels, names)]
        return LightResult(rt, *axes)

    def _print_serial(self, rt, labels, names, basename, header,
                      rowlabel, skyfill):
        flabels = self.fmt_names(names[-1], labels[-1])
        rowlabels = self.row_labels(labels[0])
        if skyfill is not None:
            self._check_sky_data(names[0], skyfill, rowlabels)
        for i, j in enumerate(flabels):
            if skyfill:
                data = skyfill.fill_data(rt[..., i])
                rowlabels = self.row_labels(skyfill.rowlabel)
            else:
                data = rt[..., i]
            f = open(f"{basename}_{j}_{names[-2]}.txt", 'w')
            self._print(f, data, header, rowlabels, rowlabel)
            f.close()

    def print_serial(self, col, basename, header=True,
                     rowlabel=True, skyfill=None, **kwargs):
        """print 3d result to series of 2d files
        """
        rt, labels, names = self.pull(*col, preserve=2, **kwargs)
        if header:
            header = self.pull_header(names[0:-1], labels[0:-1], rowlabel)
        self._print_serial(rt, labels, names, basename, header, rowlabel,
                           skyfill)

    def _pull2hdr_kdplan(self, pm, basename, rt, flabels0, flabels1, res=480):
        img, vecs, mask, mask2, header = pm.init_img(res)
        pts = self.axis("point").value_array()
        kd = cKDTree(pts)
        err, idx = kd.query(vecs[mask])
        for i, la0 in enumerate(flabels0):
            data = rt[..., i]
            for k, (la, d) in enumerate(zip(flabels1, data.T)):
                img[mask] = d[idx]
                io.array2hdr(img, f"{basename}_{la}_{la0}.hdr", [header])

    def rebase(self, points):
        paxis = ResultAxis(points, "point")
        omet = self.axis("metric").values
        mf = [i for i, v in enumerate(omet) if v not in ('x', 'y', 'z', 'area')]
        maxis = ResultAxis([omet[i] for i in mf] + ["rebase_err"], "metric")

        pts = self.axis("point").value_array()
        kd = cKDTree(pts)
        err, idx = kd.query(points)
        odata = self.data[:, idx][..., mf]
        oerr = np.broadcast_to(err[:, None, None], (*odata.shape[:-1], 1))
        odata = np.concatenate((odata, oerr), axis=-1)
        lr = LightResult(np.stack(odata), self.axes[0], paxis, self.axes[2],
                         maxis)
        return lr

    @staticmethod
    def _pull2hdr_sky(skyfill, basename, spd, rt, flabels0, flabels1):
        if skyfill.skydata.shape[0] % 365 == 0:
            gshp = (365, int(skyfill.skydata.shape[0]/365))
            hour = np.arange(skyfill.skydata.shape[0])
            gsort = np.lexsort((-np.mod(hour, gshp[1]), np.floor(hour/gshp[1])))
            psize = (2, max(5, int(240/gshp[1])))
        elif skyfill.skydata.shape[0] % spd == 0:
            gshp = (int(skyfill.skydata.shape[0]/spd), spd)
            hour = np.arange(skyfill.skydata.shape[0])
            gsort = np.lexsort((-np.mod(hour, spd), np.floor(hour/spd)))
            psize = (int(730/gshp[0]), max(5, int(240/gshp[1])))
        else:
            raise ValueError(f"skydata must have 365 days and/or {spd} "
                             f"steps per day")
        for i, la0 in enumerate(flabels0):
            if skyfill:
                data = skyfill.fill_data(rt[..., i])
            else:
                data = rt[..., i]
            for k, (la, d) in enumerate(zip(flabels1, data.T)):
                do = d[gsort].reshape(gshp)
                do = np.repeat(np.repeat(do, psize[1], 1), psize[0], 0)
                io.array2hdr(do[-1::-1], f"{basename}_{la}_"
                                         f"{la0}.hdr")

    def pull2planhdr(self, imgzone, basename, showsample=False, res=480,
                     **kwargs):
        pm = PlanMapper(imgzone)
        rt, labels, names = self.pull("metric", preserve=2, **kwargs)
        flabels0 = self.fmt_names(names[-1], labels[-1])
        flabels1 = self.fmt_names(names[-2], labels[-2])
        return self._pull2hdr_kdplan(pm, basename, rt, flabels0, flabels1,
                                     res=res)

    def pull2hdr(self, col, basename, skyfill=None, spd=24, pm=None, res=480,
                 **kwargs):
        rt, labels, names = self.pull(*col, preserve=2, **kwargs)
        flabels0 = self.fmt_names(names[-1], labels[-1])
        flabels1 = self.fmt_names(names[-2], labels[-2])
        if "sky" in names[-1]:
            pts = self.axis("point").values
            if rt.shape[0] != len(pts):
                raise ValueError(f"cannot grid {rt.shape[0]} to {len(pts)} "
                                 "points. make sure non-point axes besides "
                                 "'col' are filtered to a single value")
            if pm is None:
                pm = PlanMapper(self.axis("point").value_array()[:, 0:3])
            return self._pull2hdr_kdplan(pm, basename, rt, flabels0, flabels1,
                                         res=res)
        if skyfill is None:
            raise ValueError("'pull2hdr' with 'sky' requires skyfill")
        elif rt.shape[0] != skyfill.smtx.shape[0]:
            raise ValueError("SkyData and result do not have the same number "
                             f"of values ({skyfill.smtx.shape[0]} vs. "
                             f"{rt.shape[0]}) check that SkyData matches result"
                             ", and is appropriately masked, or that non-sky "
                             "axes besides 'col' are filtered to a single "
                             "value")
        return self._pull2hdr_sky(skyfill, basename, spd, rt, flabels0, flabels1)

    def _check_sky_data(self, col, skydata, rowlabels):
        if len(self.data.shape) == 2:
            raise ValueError("skyfill only compatible with 4d lightresults")
        skysize = self.axis("sky").values.size
        if skydata.daysteps != skysize:
            raise ValueError(f"LightResult ({skysize}) and SkyData "
                             f"({skydata.daysteps}) do not match along sky "
                             f"axis")
        fv = "\t".join(["0"]*len(rowlabels[0].split("\t")))
        if len(rowlabels) != skydata.smtx.shape[0]:
            raise ValueError(f"pulled data has {len(rowlabels)} rows but "
                             f"{skydata.smtx.shape[0]} rows expected by "
                             f"SkyData")
        return skydata.fill_data(np.asarray(rowlabels), fv)

    def info(self):
        caps = {'default': 20, 'metric': 100}
        ns = self.names
        sh = self.data.shape
        axs = self.axes
        infostr = [f"{type(self).__name__} {self.file}:",
                   f"Has {len(ns)} axes: {ns}"]
        for n, s, a in zip(ns, sh, axs):
            infostr.append(f"  Axis '{n}' has length {s}:")
            v = a.values
            try:
                cap = caps[n]
            except KeyError:
                cap = caps['default']
            if len(v) <= cap:
                for i, k in enumerate(v):
                    infostr.append(f"  {i: 5d} {k}")
            else:
                for i in [0, 1, 2, 3, 4, "...",
                          s - 5, s - 4, s - 3, s - 2, s - 1]:
                    if i == "...":
                        infostr.append(f"  {i}")
                    else:
                        infostr.append(f"  {i: 5d} {v[i]}")
        return "\n".join(infostr)

    def _index(self, i):
        """interpret indice as an axes key or a range index"""
        try:
            # look up by name
            return self.names.index(i)
        except ValueError:
            pass
        try:
            # look up by index
            if i < len(self.names):
                return i
            raise IndexError("index out of range")
        except TypeError:
            raise IndexError(f"invalid index, axis '{i}' is not in LightResult")
