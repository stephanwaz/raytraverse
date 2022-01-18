# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
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

    def __init__(self, data, *axes, header=None):
        self._file = None
        if not hasattr(data, "shape"):
            self._file = data
            data, axes, header = self.load(data)
        if len(data.shape) != len(axes):
            raise ValueError(f"data of shape: {data.shape} requires "
                             f"{len(data.shape)} axes arguments.")
        self._data = data
        self._axes = axes
        self._names = [a.name for a in axes]
        self._header = header

    @property
    def data(self):
        return self._data

    @property
    def axes(self):
        return self._axes

    @property
    def names(self):
        return self._names

    @property
    def header(self):
        return self._header

    @property
    def file(self):
        return self._file

    def axis(self, name):
        return self.axes[self.names.index(name)]

    @staticmethod
    def load(file):
        with np.load(file) as result:
            data = result['data']
            names = result['names']
            axes = tuple([ResultAxis(result[f"arr_{i}"], n)
                          for i, n in enumerate(names)])
            try:
                header = result['heeader']
            except KeyError:
                header = None
        return data, axes, header

    def write(self, file, compressed=True):
        kws = dict(data=self.data, names=self.names, header=self.header)
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

    def pull(self, *axes, aindices=None, findices=None, order=None):
        """arrange and extract data slices from result.

        Integrators construct a light result with these axes:

            0. sky
            1. point
            2. view
            3. metric

        Parameters
        ----------
        axes: Union[int, str]
            the axes (by name or integer index) to maintain and order
            the returned result (where axes will be the last N axes of result)
        aindices: Sequence[array_like], optional
            sequence of returned axis indices, up to one per each of axes to
            return a subset of data along these axes.
        findices: Sequence, optional
            sequence of indices or slices for pre-flattened axes to be flattened. give
            in order matching "order"
        order: Sequence
            the remainder of the axes in the order in which they should be
            arranged prior to flattening. by default uses their original order
            in self.data
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
        if aindices is None:
            aindices = []
        elif len(axes) == 1:
            aindices = np.atleast_2d(aindices)
        # get the indexes and shape of keeper axes
        idx = [self._index(i) for i in axes]
        shp = np.array(self.data.shape)[idx]

        # make order or remaining axes and check for errors
        if order is None:
            order = [i for i in range(len(self.names)) if i not in idx]
        elif len(order) != len(self.names) - len(axes):
            raise ValueError("axes + order must include all axes of data, "
                             "give each axes index exactly once.")
        else:
            order = [self._index(i) for i in order]
        if len(set(order + idx)) != len(order + idx):
            raise ValueError("axes + order cannot include duplicate axes, "
                             "give each axes index exactly once.")
        if findices is None:
            findices = []
        oshp = []
        ax0 = []
        fi2 = []
        for i in range(len(order)):
            if i < len(findices):
                d0 = self.axes[order[i]].values
                d = d0[findices[i]]
                fi2.append(np.arange(len(d0))[findices[i]])
                oshp.append(range(len(d)))
                ax0.append(d)
            else:
                oshp.append(range(self.data.shape[order[i]]))
                ax0.append(self.axes[order[i]].values)
        # make index values for flattened axis
        ij = np.meshgrid(*oshp, indexing='ij')

        ax0 = [ax[i.ravel()] for i, ax in zip(ij, ax0)]
        ax0_name = "_".join([self.names[i] for i in order])

        labels = [list(zip(*ax0))] + [self.axes[i].values for i in idx]
        names = [ax0_name] + [self.names[i] for i in idx]
        data = self.data
        for i, slc in zip(order, fi2):
            data = np.take(data, slc, axis=i)
        # transpose result and apply slice
        result = np.transpose(data, order + idx).reshape(-1, *shp)
        for i, slc in enumerate(aindices):
            if slc is not None:
                result = np.take(result, slc, axis=i+1)
                labels[i+1] = labels[i+1][slc]

        return result, labels, names

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
    def pull_header(names, labels, rowlabel=True):
        h = [str(i) for i in labels[1]]
        if rowlabel:
            # construct row label format
            row_label_names = dict(sky="sky", point="x\ty\tz",
                                   view="dx\tdy\tdz", image="image",
                                   metric="metric")
            h = [row_label_names[i] for i in names[0].split("_")] + h
        return "\t".join(h)

    def print(self, col, aindices=None, findices=None, order=None,
              header=True, rowlabel=True, file=None, skyfill=None):
        """first calls pull and then prints 2d result to file"""
        rt, labels, names = self.pull(col, aindices=aindices,
                                      findices=findices, order=order)
        if header:
            header = self.pull_header(names, labels, rowlabel)
        rowlabels = self.row_labels(labels[0])
        if skyfill is not None:
            rowlabels = self._check_sky_data(col, skyfill, rowlabels)
            rt = skyfill.fill_data(rt)
        self._print(file, rt, header, rowlabels, rowlabel)

    def print_serial(self, col, basename, aindices=None, findices=None,
                     order=None, header=True, rowlabel=True, skyfill=None):
        """print 3d result to series of 2d files

            col[0] is column, col[1] is file
        """
        rt, labels, names = self.pull(*col, aindices=aindices,
                                      findices=findices, order=order)
        if aindices[1] is None:
            idxs = range(rt.shape[-1])
        else:
            idxs = aindices[1]
        if header:
            header = self.pull_header(names[0:-1], labels[0:-1], rowlabel)
        rowlabels = self.row_labels(labels[0])
        if skyfill is not None:
            rowlabels = self._check_sky_data(col, skyfill, rowlabels)
        for i, j in enumerate(idxs):
            f = open(f"{basename}_{names[-1]}_{j:04d}.txt", 'w')
            if skyfill:
                data = skyfill.fill_data(rt[..., i])
            else:
                data = rt[..., i]
            self._print(f, data, header, rowlabels, rowlabel)
            f.close()

    def pull2hdr(self, col, basename, aindices=None, findices=None, order=None,
                 skyfill=None):
        rt, labels, names = self.pull(*col, aindices=aindices,
                                      findices=findices, order=order)
        pts = self.axis(order[0]).values
        if order[0] not in ['point', 'sky'] or len(pts) != rt.shape[0]:
            raise ValueError("'pull2hdr' is only compatible over points or sky "
                             "and a single element in the last dimension")
        if order[0] == 'sky' and skyfill is None:
            raise ValueError("'pull2hdr' with 'sky' requires skyfill")
        if aindices[1] is None:
            idxs = range(rt.shape[-1])
        else:
            idxs = aindices[1]
        if order[0] == 'point':
            pts = np.array(list(zip(*pts))).T
            gshp = (np.unique(pts[:, 0]).size, np.unique(pts[:, 1]).size)
            gsort = np.lexsort((pts[:, 1], pts[:, 0]))
            psize = int(500/max(gshp)/2)*2
            psize = (psize, psize)
        else:
            gshp = (365, 24)
            hour = np.arange(8760)
            gsort = np.lexsort((-np.mod(hour, 24), np.floor(hour/24)))
            psize = (2, 10)
        for i, j in enumerate(idxs):
            if skyfill:
                data = skyfill.fill_data(rt[..., i])
            else:
                data = rt[..., i]
            for k, (la, d) in enumerate(zip(labels[1], data.T)):
                do = d[gsort].reshape(gshp)
                do = np.repeat(np.repeat(do, psize[1], 1), psize[0], 0)
                if col[0] == 'metric':
                    lab = la
                else:
                    lab = f"{col[0]}_{k:04d}"
                io.array2hdr(do[-1::-1], f"{basename}_{order[0]}_{lab}_"
                                         f"{names[-1]}_{j:04d}.hdr")

    def pull2pandas(self, ax1, ax2, **kwargs):
        """returns a list of dicts suitable for initializing pandas.DataFrames

        Parameters
        ----------
        ax1: Union[int, str]
            the output row axis
        ax2: Union[int, str]
            the output column axis
        kwargs: dict
            additional parameters for self.pull()
        Returns
        -------
        panda_args: Sequence[dict]
            list of keyword arguments for initializing a pandas DataFrame::

                frames = [pandas.DataFrame(**kw) for kw in panda_args]

            keys are ['data', 'index', 'columns']
        frame_info: Sequence[dict]
            information for each data frame keys:

                - name: the summary name of the frame, a concatenation of the
                  flattened axes (for example: "point_view" implies the frame is
                  extracted for a particular point and view direction)
                - item: the values of the frame from each of the flatten axes
                  (for example: for a name "point_view" this
                  item = [(x, y, z), (vx, vy, vz)]
                - axis0: the name of the row axis (for example: "sky")
                - axis1: the name of thee column axis (for example: "metric")

        """
        result, labels, names = self.pull(ax1, ax2, **kwargs)
        for i in range(len(labels)):
            try:
                if len(labels[i].shape) > 1:
                    labels[i] = [tuple(j) for j in labels[i]]
            except AttributeError:
                pass
        panda_args = []
        frame_info = []
        for r, l in zip(result, labels[0]):
            panda_args.append(dict(data=r, index=labels[1], columns=labels[2]))
            frame_info.append(dict(name=names[0], item=l, axis0=names[1],
                                   axis1=names[2]))
        return panda_args, frame_info

    def _check_sky_data(self, col, skydata, rowlabels):
        if col == "sky":
            raise ValueError("skyfill cannot be used with col='sky'")
        if len(self.data.shape) == 2:
            raise ValueError("skyfill only compatible with 4d lightresults")
        skysize = self.axes[self._index("sky")].values.size
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
        infostr = [f"LightResult {self.file}:", f"{self.header}",
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
            raise IndexError("invalid index")
