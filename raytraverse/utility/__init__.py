# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================

"""utilities"""
__all__ = ['pool_call', 'TStqdm', 'imagetools']

from raytraverse.utility.tstqdm import TStqdm


def pool_call(func, args, *fixed_args, cap=None, desc="processing", **kwargs):
    """calls func for a sequence of arguments using a ProcessPool executor
    and a progress bar. result is equivalent to:

    |    result = []
    |    for arg in args:
    |        result.append(func(*args, *fixed_args, **kwargs))
    |    return result

    Parameters
    ----------
    func: callable
        the function to execute in parallel
    args: Sequence[Sequence]
        list of arguments (each item is expanded with *). first N args of func
    fixed_args: Sequence
        arguments passed to func that are the same for all calls (next N
        arguments  after args)
    cap: int, optional
        execution cap for ProcessPool
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
        done = set()
        not_done = set()
        cnt = 0
        pbar_t = 0
        # submit asynchronous to process pool
        for arg in args:
            # manage to queue to avoid loading too many points in memory
            # and update progress bar as completed
            if cnt > pbar.nworkers*3:
                wait_r = pbar.wait(not_done, return_when=pbar.FIRST_COMPLETED)
                not_done = wait_r.not_done
                done.update(wait_r.done)
                pbar.update(len(done) - pbar_t)
                pbar_t = len(done)
            fu = exc.submit(func, *arg, *fixed_args, **kwargs)
            futures.append(fu)
            not_done.add(fu)
            cnt += 1
        # gather results (in order)
        for future in futures:
            results.append(future.result())
            if future in not_done:
                pbar.update(1)
    return results
