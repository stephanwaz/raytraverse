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

static void rcprintdefaults(void);
extern int rcontrib_init(int  argc, char  *argv[]);
extern void rcontrib_loadscene(char* octname);
extern void rcontrib_call(char *fname);
extern void rcontrib_clear(void);
static FILE* rcinit2(char *fname);
extern void parental_loop2(char *fname);
extern void feeder_loop2(char *fname);
extern int in_rchild2(void);
extern int getvecfp(FVECT vec, FILE *fp);
extern int outbright;

#ifdef __cplusplus
}
#endif
#endif /* _RAD_RTMAIN_H_ */

