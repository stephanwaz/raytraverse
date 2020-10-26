#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
# =======================================================================
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
# =======================================================================


"""entry point for genskyvec."""
import subprocess
import sys


def main():
    sys.argv[0] = __file__.rsplit('/', 1)[0] + '/genskyvec.pl'
    subprocess.call(sys.argv)


if __name__ == '__main__':
    main()
