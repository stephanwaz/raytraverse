/* Copyright (c) 2020 Stephen Wasilewski, HSLU and EPFL
 * =======================================================================
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 *=======================================================================
 *
 * The Following code is copied from, adapts, includes, and/or links parts of
 * the Radiance source code which is licensed by the following:
 *
 * The Radiance Software License, Version 1.0
 *
 * Copyright (c) 1990 - 2018 The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory.   All rights reserved.
 *
 * If a copy of the full text of the License was not distributed with this file
 * (in ./ray/License.txt) the License is available at
 * https://www.radiance-online.org/download-install/license
 */

#ifndef _RAD_RTINIT_H_
#define _RAD_RTINIT_H_

#ifdef __cplusplus
extern "C" {
#endif

extern int return_value_count;

static void onsig(int  signo);
static void sigdie(int  signo, char  *msg);
extern int rtinit(int  argc, char  **argv);
extern void rtrace_loadscene(char* octname);
extern int rtrace_loadsrc(char* srcname, int freesrc);
extern int setoutput2(char *vs);
extern void rtrace_setup(int nproc);
extern RREAL* rtrace_call(double *vptr, int nproc, int raycount);
void oputrad(RAY  *r);

#ifdef __cplusplus
}
#endif
#endif /* _RAD_RTMAIN_H_ */

