# -*- coding: utf-8 -*-

# Copyright (c) 2021 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
import os

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
                   " not first in order and all but first reduces to 1."
                   " LightResult should be a full evaluation (not masked)"),
 click.option("--header/--no-header", default=True, help="print col labels"),
 click.option("--rowlabel/--no-rowlabel", default=True, help="label row"),
 click.option("--info/--no-info", default=False,
              help="skip execution and return shape and axis info about "
                   "LightResult"),
 click.option("--gridhdr/--no-gridhdr", default=False,
              help="use with 'ofiles', order 'X point/sky Y' and make sure Y"
                   " only has one value (with appropriate filter)"),
 ]


def shared_pull(ctx, lr=None, col="metric", order=('point', 'view', 'sky'),
                ofiles=None, ptfilter=None, viewfilter=None, skyfilter=None,
                imgfilter=None, metricfilter=None, skyfill=None, header=True,
                rowlabel=True, info=False, gridhdr=False, **kwargs):
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
        click.echo(result.info(), err=True)
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
        if skyfilter is not None:
            skydata.mask = skyfilter
    else:
        skydata = None

    if ofiles is None:
        result.print(col, aindices=filters[col], findices=findices, order=order,
                     skyfill=skydata, **pargs)
    else:
        if hasattr(findices[0], "stop"):
            findices[0] = None
        aindices = [filters[col], findices[0]]
        findices = findices[1:]
        col = [col, order[0]]
        order = order[1:]
        if gridhdr:
            result.pull2hdr(col, ofiles, aindices=aindices, findices=findices,
                            order=order, skyfill=skydata)
        else:
            result.print_serial(col, ofiles, aindices=aindices, findices=findices,
                                order=order, skyfill=skydata, **pargs)
