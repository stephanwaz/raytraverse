# -*- coding: utf-8 -*-

# Copyright (c) 2021 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import os
import sys

import numpy as np
from clasp import click
import clasp.click_ext as clk

from raytraverse.lightfield import LightResult
from raytraverse.sky import SkyData


@clk.pretty_name("NPY, TSV, FLOATS,FLOATS")
def np_load(ctx, param, s):
    """read np array from command line

    trys np.load (numpy binary), then np.loadtxt (space seperated txt file)
    then split row by spaces and columns by commas.
    """
    if s is None:
        return s
    if s == '-':
        s = clk.tmp_stdin(ctx)
    if os.path.exists(s):
        try:
            ar = np.load(s)
        except ValueError:
            ar = np.loadtxt(s)
        if len(ar.shape) == 1:
            ar = ar.reshape(1, -1)
        return ar
    else:
        return np.array([[float(i) for i in j.split(',')] for j in s.split()])


@clk.pretty_name("NPY, TSV, FLOATS,FLOATS, FILE")
def np_load_safe(ctx, param, s):
    try:
        return np_load(ctx, param, s)
    except ValueError as ex:
        if os.path.exists(s):
            return s
        else:
            raise ex


pull_decs = [
 click.option("-lr", callback=clk.is_file,
              help=".npz LightResult, overrides lightresult from chained "
                   "commands (evaluate/imgmetric). required if not chained "
                   "with evaluate or imgmetric."),
 click.option("-col", default='metric',
              type=click.Choice(['metric', 'point', 'view', 'sky']),
              help="axis to preserve"),
 click.option("-order", default="point view sky", callback=clk.split_str,
              help="order for flattening remaining result axes. Note that"
                   " in the case of an imgmetric result, this option is ignored"
                   " as the result is already 2D"),
 click.option("-ofiles",
              help="if given output serialized files along first axis "
                   "(given by order) with naming [ofiles]_XXXX.txt"),
 click.option("-ptfilter", callback=clk.split_int,
              help="point indices to return (ignored for imgmetric result)"),
 click.option("-viewfilter", callback=clk.split_int,
              help="view direction indices to return "
                   "(ignored for imgmetric result)"),
 click.option("-skyfilter", callback=clk.split_int,
              help="sky indices to return (ignored for imgmetric result)"),
 click.option("-imgfilter", callback=clk.split_int,
              help="image indices to return (ignored for lightfield result)"),
 click.option("-metricfilter", callback=clk.split_str,
              help="metrics to return (non-existant are ignored)"),
 click.option("-skyfill", callback=clk.is_file,
              help="path to skydata file. assumes rows are timesteps."
                   " skyfilter should be None and other beside col "
                   "should reduce to 1 or ofiles is given and sky is"
                   " not first in order and all but first reduces to 1"
                   " LightResult should be a full evaluation (not masked)"),
 click.option("--header/--no-header", default=True, help="print col labels"),
 click.option("--rowlabel/--no-rowlabel", default=True, help="label row"),
 click.option("--info/--no-info", default=False,
              help="skip execution and return shape and axis info about "
                   "LightResult")
 ]


def shared_pull(ctx, lr=None, col="metric", order=('point', 'view', 'sky'),
                ofiles=None, ptfilter=None, viewfilter=None, skyfilter=None,
                imgfilter=None, metricfilter=None, skyfill=None, header=True,
                rowlabel=True, info=False, **kwargs):
    """used by both raytraverse.cli and raytu, add pull_decs and
    clk.command_decs as  clk.shared_decs in main script so click can properly
    load options"""
    if lr is not None:
        result = LightResult(lr)
    elif 'lightresult' in ctx.obj:
        result = ctx.obj['lightresult']
    else:
        click.echo("Please provide an -lr option (path to light result file)",
                   err=True)
        raise click.Abort
    if info:
        ns = result.names
        sh = result.data.shape
        axs = result.axes
        click.echo(f"LightResult: {result.file}:", err=True)
        click.echo(result.header, err=True)
        click.echo(f"LightResult has {len(ns)} axes: {ns}", err=True)
        for n, s, a in zip(ns, sh, axs):
            click.echo(f"  Axis '{n}' has length {s}:", err=True)
            v = a.values
            if len(v) < 20:
                for i, k in enumerate(v):
                    click.echo(f"  {i: 5d} {k}", err=True)
            else:

                for i in [0, 1, 2, 3, "...", s-2, s-1]:
                    if i == "...":
                        click.echo(f"  {i}", err=True)
                    else:
                        click.echo(f"  {i: 5d} {v[i]}", err=True)
        return None
    filters = dict(metric=metricfilter, sky=skyfilter, point=ptfilter,
                   view=viewfilter, image=imgfilter)
    # translate metric names to indices
    axes = [i.name for i in result.axes]
    if metricfilter is not None:
        ai = axes.index("metric")
        try:
            metricfilter = [int(i) for i in metricfilter]
        except ValueError:
            av = result.axes[ai].values
        else:
            av = list(range(len(result.axes[ai].values)))
        aindices = np.flatnonzero([i in metricfilter for i in av])
        [click.echo(f"Warning! {i} not in LightResult", err=True) for i in
         metricfilter if i not in av]
        filters["metric"] = aindices
    # translate sky index to skydata shape
    if skyfilter is not None and "sky" in axes:
        ai = axes.index("sky")
        av = result.axes[ai].values
        aindices = np.flatnonzero(np.isin(av, skyfilter))
        filters["sky"] = aindices
    if len(result.data.shape) == 2:
        order = None
        findices = [slice(None) if imgfilter is None else imgfilter]
    else:
        findices = [slice(filters[x]) if filters[x] is None else filters[x]
                    for x in order]
    pargs = dict(header=header, rowlabel=rowlabel)

    if skyfill is not None:
        skydata = SkyData(skyfill)
        if col == "sky":
            raise ValueError("skyfill cannot be used with col='sky'")
        if len(result.data.shape) == 2:
            raise ValueError("skyfill only compatible with 4d lightresults")
        skysize = result.axes[axes.index("sky")].values.size
        if skydata.daysteps != skysize:
            raise ValueError(f"LightResult ({skysize}) and SkyData "
                             f"({skydata.daysteps}) do not match along sky "
                             f"axis")
        if skyfilter is not None:
            skydata.mask = skyfilter
        if ofiles is None:
            data, hdr, rowlabels = result.return2d(col, filters[col],
                                                   findices, order, rowlabel)
            frames = {"stdout": data}
        else:
            if hasattr(findices[0], "stop"):
                findices[0] = None
            aindices = [filters[col], findices[0]]
            findices = findices[1:]
            col = [col, order[0]]
            order = order[1:]
            frames, hdr, rowlabels = result.return_serial(col, ofiles, aindices,
                                                          findices, order,
                                                          rowlabel)
        if len(rowlabels) != skydata.smtx.shape[0]:
            raise ValueError(f"pulled data has {len(rowlabels)} rows but "
                             f"{skydata.smtx.shape[0]} rows expected by "
                             f"SkyData")
        fv = "\t".join(["0"] * len(rowlabels[0].split("\t")))
        rhfull = skydata.fill_data(np.asarray(rowlabels), fv)
        for out, data in frames.items():
            if out == "stdout":
                f = sys.stdout
            else:
                f = open(out, 'w')
            dfull = skydata.fill_data(data)
            if header:
                print(hdr, file=f)
            for rh, d in zip(rhfull, dfull):
                rl = "\t".join([f"{i:.05f}" for i in d])
                if rowlabel:
                    rl = rh + "\t" + rl
                print(rl, file=f)

    elif ofiles is None:
        result.print(col, aindices=filters[col], findices=findices, order=order,
                     **pargs)
    else:
        if hasattr(findices[0], "stop"):
            findices[0] = None
        aindices = [filters[col], findices[0]]
        findices = findices[1:]
        col = [col, order[0]]
        order = order[1:]
        result.print_serial(col, ofiles, aindices=aindices, findices=findices,
                            order=order, **pargs)
