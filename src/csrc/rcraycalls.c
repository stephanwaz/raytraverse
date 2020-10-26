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

/*
 *  rcrayalls.c - modified raycalls.c, removes redundant declarations for compiling with rcontrib.c.
 */

#include "copyright.h"
#include <string.h>
#include <time.h>

#include  "ray.h"
#include  "source.h"
#include  "bsdf.h"
#include  "ambient.h"
#include  "otypes.h"
#include  "random.h"
#include  "data.h"
#include  "font.h"
#include  "pmapray.h"

char	*progname = "unknown_app";	/* caller sets to argv[0] */

char	*octname;			/* octree name we are given */
int	dimlist[MAXDIM];		/* sampling dimensions */
char	*amblist[AMBLLEN+1];		/* ambient include/exclude list */


static void
reset_random(void)		/* re-initialize random number generator */
{
	if (rand_samp) {
		srandom((long)time(0));
		initurand(0);
	} else {
		srandom(0L);
		initurand(2048);
	}
}


void
ray_init(			/* initialize ray-tracing calculation */
	char	*otnm
)
{
	if (nobjects > 0)		/* free old scene data */
		ray_done(0);
					/* initialize object types */
	if (ofun[OBJ_SPHERE].funp == o_default)
		initotypes();
					/* initialize urand */
	reset_random();
					/* read scene octree */
	readoct(octname = otnm, ~(IO_FILES|IO_INFO), &thescene, NULL);
	nsceneobjs = nobjects;
					/* PMAP: Init & load photon maps */
	ray_init_pmap();
					/* find and mark sources */
	marksources();
					/* initialize ambient calculation */
	setambient();
					/* ready to go... (almost) */
}


void
ray_trace(			/* trace a primary ray */
	RAY	*r
)
{
	rayorigin(r, PRIMARY, NULL, NULL);
	samplendx++;
	rayvalue(r);		/* assumes origin and direction are set */
}


void
ray_done(		/* free ray-tracing data */
	int	freall
)
{
	retainfonts = 1;
	ambdone();
	ambnotify(OVOID);
	freesources();
	freeobjects(0, nobjects);
	donesets();
	octdone();
	thescene.cutree = EMPTY;
	octname = NULL;
	retainfonts = 0;
	if (freall) {
		freefont(NULL);
		freedata(NULL);
		SDfreeCache(NULL);
		initurand(0);
	}
	if (nobjects > 0) {
		sprintf(errmsg, "%ld objects left after call to ray_done()",
				(long)nobjects);
		error(WARNING, errmsg);
	}

	ray_done_pmap();
}


void
ray_save(			/* save current parameter settings */
	RAYPARAMS	*rp
)
{
	int	i, ndx;

	if (rp == NULL)
		return;
	rp->do_irrad = do_irrad;
	rp->rand_samp = rand_samp;
	rp->dstrsrc = dstrsrc;
	rp->shadthresh = shadthresh;
	rp->shadcert = shadcert;
	rp->directrelay = directrelay;
	rp->vspretest = vspretest;
	rp->directvis = directvis;
	rp->srcsizerat = srcsizerat;
	copycolor(rp->cextinction, cextinction);
	copycolor(rp->salbedo, salbedo);
	rp->seccg = seccg;
	rp->ssampdist = ssampdist;
	rp->specthresh = specthresh;
	rp->specjitter = specjitter;
	rp->backvis = backvis;
	rp->maxdepth = maxdepth;
	rp->minweight = minweight;
	if (ambfile != NULL)
		strncpy(rp->ambfile, ambfile, sizeof(rp->ambfile)-1);
	else
		memset(rp->ambfile, '\0', sizeof(rp->ambfile));
	copycolor(rp->ambval, ambval);
	rp->ambvwt = ambvwt;
	rp->ambacc = ambacc;
	rp->ambres = ambres;
	rp->ambdiv = ambdiv;
	rp->ambssamp = ambssamp;
	rp->ambounce = ambounce;
	rp->ambincl = ambincl;
	memset(rp->amblval, '\0', sizeof(rp->amblval));
	ndx = 0;
	for (i = 0; i < AMBLLEN && amblist[i] != NULL; i++) {
		int	len = strlen(amblist[i]);
		if (ndx+len >= sizeof(rp->amblval))
			break;
		strcpy(rp->amblval+ndx, amblist[i]);
		rp->amblndx[i] = ndx;
		ndx += len+1;
	}
	while (i <= AMBLLEN)
		rp->amblndx[i++] = -1;

	/* PMAP: save photon mapping params */
	ray_save_pmap(rp);
}


void
ray_restore(			/* restore parameter settings */
	RAYPARAMS	*rp
)
{
	int	i;

	if (rp == NULL) {		/* restore defaults */
		RAYPARAMS	dflt;
		ray_defaults(&dflt);
		ray_restore(&dflt);
		return;
	}
					/* restore saved settings */
	do_irrad = rp->do_irrad;
	if (!rand_samp != !rp->rand_samp) {
		rand_samp = rp->rand_samp;
		reset_random();
	}
	dstrsrc = rp->dstrsrc;
	shadthresh = rp->shadthresh;
	shadcert = rp->shadcert;
	directrelay = rp->directrelay;
	vspretest = rp->vspretest;
	directvis = rp->directvis;
	srcsizerat = rp->srcsizerat;
	copycolor(cextinction, rp->cextinction);
	copycolor(salbedo, rp->salbedo);
	seccg = rp->seccg;
	ssampdist = rp->ssampdist;
	specthresh = rp->specthresh;
	specjitter = rp->specjitter;
	backvis = rp->backvis;
	maxdepth = rp->maxdepth;
	minweight = rp->minweight;
	copycolor(ambval, rp->ambval);
	ambvwt = rp->ambvwt;
	ambdiv = rp->ambdiv;
	ambssamp = rp->ambssamp;
	ambounce = rp->ambounce;
					/* a bit dangerous if not static */
	for (i = 0; rp->amblndx[i] >= 0; i++)
		amblist[i] = rp->amblval + rp->amblndx[i];
	while (i <= AMBLLEN)
		amblist[i++] = NULL;
	ambincl = rp->ambincl;
					/* update ambient calculation */
	ambnotify(OVOID);
	if (thescene.cutree != EMPTY) {
		int	newamb = (ambfile == NULL) ?  rp->ambfile[0] :
					strcmp(ambfile, rp->ambfile) ;

		if (amblist[0] != NULL)
			for (i = 0; i < nobjects; i++)
				ambnotify(i);

		ambfile = (rp->ambfile[0]) ? rp->ambfile : (char *)NULL;
		if (newamb) {
			ambres = rp->ambres;
			ambacc = rp->ambacc;
			setambient();
		} else {
			setambres(rp->ambres);
			setambacc(rp->ambacc);
		}
	} else {
		ambfile = (rp->ambfile[0]) ? rp->ambfile : (char *)NULL;
		ambres = rp->ambres;
		ambacc = rp->ambacc;
	}

	/* PMAP: restore photon mapping params */
	ray_restore_pmap(rp);
}


void
ray_defaults(		/* get default parameter values */
	RAYPARAMS	*rp
)
{
	int	i;

	if (rp == NULL)
		return;

	rp->do_irrad = 0;
	rp->rand_samp = 1;
	rp->dstrsrc = 0.0;
	rp->shadthresh = .03;
	rp->shadcert = .75;
	rp->directrelay = 2;
	rp->vspretest = 512;
	rp->directvis = 1;
	rp->srcsizerat = .2;
	setcolor(rp->cextinction, 0., 0., 0.);
	setcolor(rp->salbedo, 0., 0., 0.);
	rp->seccg = 0.;
	rp->ssampdist = 0.;
	rp->specthresh = .15;
	rp->specjitter = 1.;
	rp->backvis = 1;
	rp->maxdepth = -10;
	rp->minweight = 2e-3;
	memset(rp->ambfile, '\0', sizeof(rp->ambfile));
	setcolor(rp->ambval, 0., 0., 0.);
	rp->ambvwt = 0;
	rp->ambres = 256;
	rp->ambacc = 0.15;
	rp->ambdiv = 1024;
	rp->ambssamp = 512;
	rp->ambounce = 0;
	rp->ambincl = -1;
	memset(rp->amblval, '\0', sizeof(rp->amblval));
	for (i = AMBLLEN+1; i--; )
		rp->amblndx[i] = -1;

	/* PMAP: restore photon mapping defaults */
	ray_defaults_pmap(rp);
}
