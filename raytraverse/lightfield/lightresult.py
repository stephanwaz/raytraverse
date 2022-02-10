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

from raytraverse import io


class ResultAxis(object):

    def __init__(self, values, name):
        self.values = np.asarray(values)
        if len(self.values.shape) == 2:
            dt = type(self.values.flat[0])
            dtype = np.dtype([(f"f{i}", dt) for i in
                              range(self.values.shape[1])])
            self.values = np.array(list(zip(*self.values.T)), dtype=dtype)
        self.name = name


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
        kwargs: dict, optional
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
            rl = "\t".join([f"{i:.05f}" for i in r])
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
                rl = "_".join(f"{i:.02f}" for i in rh)
            elif np.issubdtype(type(rh), np.integer):
                rl = f"{rh:04d}"
            else:
                rl = rh
            rls.append(f"{name}_{rl}")
        return rls

    def _set_rln(self, row_label_names, axis, labels):
        try:
            ptaxis = self.axis(axis).values[0]
        except IndexError:
            pass
        else:
            if not hasattr(ptaxis, "__len__"):
                row_label_names[axis] = axis
            else:
                ptlabel = labels[0:len(ptaxis)]
                row_label_names[axis] = "\t".join(ptlabel)

    def pull_header(self, names, labels, rowlabel=True):
        h = [str(i) for i in labels[1]]
        if rowlabel:
            # construct row label format
            row_label_names = dict(sky="sky", point="x\ty\tz",
                                   view="dx\tdy\tdz", image="image",
                                   metric="metric")
            self._set_rln(row_label_names, "point",
                          ["x", "y", "z", "dx", "dy", "dz"])
            self._set_rln(row_label_names, "view",
                          ["dx", "dy", "dz"])
            h = [row_label_names[i] for i in names[0].split("_")
                 if i in row_label_names] + h
        return "\t".join(h)

    def print(self, col, header=True, rowlabel=True, file=None,
              skyfill=None, **kwargs):
        """first calls pull and then prints 2d result to file"""
        rt, labels, names = self.pull(*col, **kwargs)
        if header:
            header = self.pull_header(names, labels, rowlabel)
        rowlabels = self.row_labels(labels[0])
        if skyfill is not None:
            rowlabels = self._check_sky_data(col[0], skyfill, rowlabels)
            rt = skyfill.fill_data(rt)
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
            rowlabels = self._check_sky_data(names[0], skyfill, rowlabels)
        for i, j in enumerate(flabels):
            if skyfill:
                data = skyfill.fill_data(rt[..., i])
            else:
                data = rt[..., i]
            f = open(f"{basename}_{j}.txt", 'w')
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

    def pull2hdr(self, col, basename, skyfill=None, spd=24,  **kwargs):
        rt, labels, names = self.pull(*col, preserve=2, **kwargs)
        if names[-1] == "sky":
            pts = self.axis("point").values
            if rt.shape[0] != len(pts):
                raise ValueError(f"cannot grid {rt.shape[0]} to {len(pts)} "
                                 "points. make sure non-point axes besides "
                                 "'col' are filtered to a single value")
            pts = np.array(list(zip(*pts))).T
            gshp = (np.unique(pts[:, 0]).size, np.unique(pts[:, 1]).size)
            gsort = np.lexsort((pts[:, 1], pts[:, 0]))
            psize = int(500/max(gshp)/2)*2
            psize = (psize, psize)
        elif skyfill is None:
            raise ValueError("'pull2hdr' with 'sky' requires skyfill")
        elif rt.shape[0] != skyfill.smtx.shape[0]:
            raise ValueError("SkyData and result do not have the same number "
                             f"of values ({skyfill.smtx.shape[0]} vs. "
                             f"{rt.shape[0]}) check that SkyData matches result"
                             ", and is appropriately masked, or that non-sky "
                             "axes besides 'col' are filtered to a single "
                             "value")
        elif skyfill.skydata.shape[0] % 365 == 0:
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
        flabels0 = self.fmt_names(names[-1], labels[-1])
        flabels1 = self.fmt_names(names[-2], labels[-2])
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

    def _check_sky_data(self, col, skydata, rowlabels):
        if "sky" in col:
            raise ValueError("skyfill cannot be used with col='sky'")
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
        infostr = [f"LightResult {self.file}:",
                   f"LightResult has {len(ns)} axes: {ns}"]
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
