#!/usr/bin/env python
import sys
import os

import numpy as np
from clasp.script_tools import sglob

print()
print("opt     MAE_Ev   MAE_DGP  MAE_UGP  MSD_Ev   MSD_DGP  MSD_UGP")
print("-"*60)
total = []
for opt in sys.argv[1:]:
    results = sglob(f"results/{opt}_v*_ryt.tsv")

    np.set_printoptions(3, suppress=True)
    
    mads = []
    msds = []
    for r in results:
        ref = f"../base1compdv/scene/refrt/" + os.path.basename(r).replace("ryt", "refrt")
        rd = np.loadtxt(r, skiprows=1)[:, [1, 2, 3]]
        refd = np.loadtxt(ref, skiprows=1)[:, [4, 5, 12]]
        mask = np.logical_and(refd[:, 1] >= 0.25, refd[:, 1] <= 0.75)
        d = rd[mask] - refd[mask]
        d[:, 0] = d[:, 0]/refd[mask, 0]
        mad = np.average(np.abs(d), axis=0)
        msd = np.average(d, axis=0)
        mads.append(mad)
        msds.append(msd)
    total.append((*np.average(mads, axis=0), *np.average(msds, axis=0)))
    print("{0} {1: >8.04f} {2: >8.04f} {3: >8.04f} {4: >8.04f} {5: >8.04f} {6: >8.04f}".format(opt, *total[-1]))
total = np.average(total, axis=0)
print("-"*60)
print("{0} {1: >8.04f} {2: >8.04f} {3: >8.04f} {4: >8.04f} {5: >8.04f} {6: >8.04f}".format("total ", *total))