# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import numpy as np


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
        if not hasattr(data, "shape"):
            data, axes = self.load(data)
        if len(data.shape) != len(axes):
            raise ValueError(f"data of shape: {data.shape} requires "
                             f"{len(data.shape)} axes arguments.")
        self._data = data
        self._axes = axes
        self._names = [a.name for a in axes]

    @property
    def data(self):
        return self._data

    @property
    def axes(self):
        return self._axes

    @property
    def names(self):
        return self._names

    @staticmethod
    def load(file):
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
            file.close()

    def pull(self, *axes, aindices=None, findices=None, order=None):
        """arrange and extract data slices from result

        Parameters
        ----------
        axes: Union[int, str]
            the axes (by name or integer index) to maintain and order
            the returned result (where axes will be the last N axes of result)
        aindices: Sequence[array_like], optional
            sequence of returned axis indices, up to one per each of axes to
            return a subset of data along these axes.
        findices: Sequence[array_like], optional
            sequence of indices for pre-flattened axes to be flattened. give
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
        else:
            aindices = np.atleast_2d(aindices)
        if findices is None:
            findices = []
        else:
            findices = np.atleast_2d(findices)
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
        oshp = []
        ax0 = []
        for i in range(len(order)):
            if i < len(findices):
                oshp.append(range(len(findices[i])))
                ax0.append(self.axes[order[i]].values[findices[i]])
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
        for i, slc in zip(order, findices):
            data = np.take(data, slc, axis=i)
        # transpose result and apply slice
        result = np.transpose(data, order + idx).reshape(-1, *shp)
        for i, slc in enumerate(aindices):
            result = np.take(result, slc, axis=i+1)
            labels[i+1] = labels[i+1][slc]

        return result, labels, names

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
