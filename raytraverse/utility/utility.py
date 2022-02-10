# -*- coding: utf-8 -*-

# Copyright (c) 2019 Stephen Wasilewski
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================
from raytraverse.utility.tstqdm import TStqdm


def pool_call(func, args, *fixed_args, cap=None, expandarg=True,
              desc="processing", **kwargs):
    """calls func for a sequence of arguments using a ProcessPool executor
    and a progress bar. result is equivalent to::

         result = []
         for arg in args:
             result.append(func(*args, *fixed_args, **kwargs))
         return result

    Parameters
    ----------
    func: callable
        the function to execute in parallel
    args: Sequence[Sequence]
        list of arguments (each item is expanded with '*' unless expandarg
        is false). first N args of func
    fixed_args: Sequence
        arguments passed to func that are the same for all calls (next N
        arguments  after args)
    cap: int, optional
        execution cap for ProcessPool
    expandarg: bool, optional
        expand args with '*' when calling func
    desc: str, optional
        label for progress bar
    kwargs:
        additional keyword arguments passed to func
    Returns
    -------
    sequence of results from func (order preserved)
    """
    results = []
    with TStqdm(workers=True, total=len(args), cap=cap,
                desc=desc) as pbar:
        exc = pbar.pool
        futures = []
        # submit asynchronous to process pool
        for arg in args:
            if expandarg:
                fu = exc.submit(func, *arg, *fixed_args, **kwargs)
            else:
                fu = exc.submit(func, arg, *fixed_args, **kwargs)
            futures.append(fu)
        # gather results (in order)
        for future in futures:
            results.append(future.result())
            pbar.update(1)
    return results
