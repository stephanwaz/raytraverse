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

#ifdef __cplusplus
extern "C" {
#endif

/*
 *  rcinit.c - modified main routine from rcmain called rcontrib_init,
 *  to load command in to memory but not execute or exit.
 */

#include "copyright.h"
#include "calcompcal.h"
#include <signal.h>
#include <time.h>
#include <rcontrib.h>
#include "random.h"
#include "source.h"
#include "ambient.h"
#include "pmapray.h"
#include "pmapcontrib.h"
#include "rcinit.h"

int	gargc;				/* global argc */
char	**gargv;			/* global argv */
char	*octname;			/* global octree name */
char	*progname;			/* global argv[0] */

char	*sigerr[NSIG];			/* signal error messages */

int	nproc = 1;			/* number of processes requested */
int	nchild = 0;			/* number of children (-1 in child) */

int	inpfmt = 'a';			/* input format */
int	outfmt = 'a';			/* output format */

int	header = 1;			/* output header? */
int outbright = 0;
int	force_open = 0;			/* truncate existing output? */
int	recover = 0;			/* recover previous output? */
int	accumulate = 1;			/* input rays per output record */
int	contrib = 0;			/* computing contributions? */

int	xres = 0;			/* horizontal (scan) size */
int	yres = 0;			/* vertical resolution */

int	using_stdout = 0;		/* are we using stdout? */

int	imm_irrad = 0;			/* compute immediate irradiance? */
int	lim_dist = 0;			/* limit distance? */

int	report_intvl = 0;		/* reporting interval (seconds) */

const char	*modname[MAXMODLIST];	/* ordered modifier name list */
int		nmods = 0;		/* number of modifiers */

//void	(*addobjnotify[8])() = {ambnotify, NULL};

char	RCCONTEXT[] = "RC.";		/* our special evaluation context */


static void
rcprintdefaults(void)			/* print default values to stdout */
{
  printf("-c %-5d\t\t\t# accumulated rays per record\n", accumulate);
  printf("-V%c\t\t\t\t# output %s\n", contrib ? '+' : '-',
         contrib ? "contributions" : "coefficients");
  if (imm_irrad)
    printf("-I+\t\t\t\t# immediate irradiance on\n");
  printf("-n %-2d\t\t\t\t# number of rendering processes\n", nproc);
  printf("-x %-9d\t\t\t# %s\n", xres,
         yres && xres ? "x resolution" : "flush interval");
  printf("-y %-9d\t\t\t# y resolution\n", yres);
  printf(lim_dist ? "-ld+\t\t\t\t# limit distance on\n" :
         "-ld-\t\t\t\t# limit distance off\n");
  printf("-h%c\t\t\t\t# %s header\n", header ? '+' : '-',
         header ? "output" : "no");
  printf("-f%c%c\t\t\t\t# format input/output = %s/%s\n",
         inpfmt, outfmt, formstr(inpfmt), formstr(outfmt));
  printf(erract[WARNING].pf != NULL ?
         "-w+\t\t\t\t# warning messages on\n" :
         "-w-\t\t\t\t# warning messages off\n");
  print_rdefaults();
}

/* set input/output format */
static void
setformat(const char *fmt)
{
  switch (fmt[0]) {
    case 'f':
    case 'd':
      SET_FILE_BINARY(stdin);
      /* fall through */
    case 'a':
      inpfmt = fmt[0];
      break;
    default:
      goto fmterr;
  }
  switch (fmt[1]) {
    case '\0':
      outfmt = inpfmt;
      return;
    case 'a':
    case 'f':
    case 'd':
    case 'c':
      outfmt = fmt[1];
      break;
    default:
      goto fmterr;
  }
  if (!fmt[2])
    return;
  fmterr:
  sprintf(errmsg, "Illegal i/o format: -f%s", fmt);
  error(USER, errmsg);
}

void
eputsrc(				/* put string to stderr */
        char  *s
)
{
  static int  midline = 0;

  if (!*s)
    return;
  if (!midline++) {
    fputs(progname, stderr);
    fputs(": ", stderr);
  }
  fputs(s, stderr);
  if (s[strlen(s)-1] == '\n') {
    fflush(stderr);
    midline = 0;
  }
}

void
wputsrc(				/* warning output function */
        char	*s
)
{
  int  lasterrno = errno;
  eputsrc(s);
  errno = lasterrno;
}

static void
onsig(				/* fatal signal */
        int  signo
)
{
  static int  gotsig = 0;

  if (gotsig++)			/* two signals and we're gone! */
    _exit(signo);

#ifdef SIGALRM
  alarm(15);			/* allow 15 seconds to clean up */
  signal(SIGALRM, SIG_DFL);	/* make certain we do die */
#endif
  eputsrc("signal - ");
  eputsrc(sigerr[signo]);
  eputsrc("\n");
  quit(3);
}


static void
sigdie(			/* set fatal signal */
        int  signo,
        char  *msg
)
{
  if (signal(signo, onsig) == SIG_IGN)
    signal(signo, SIG_IGN);
  sigerr[signo] = msg;
}


/* Set overriding options */
static void
override_options(void)
{
  shadthresh = 0;
  ambssamp = 0;
  ambacc = 0;
  if (accumulate <= 0)	/* no output flushing for single record */
    xres = yres = 0;
}


int
rcontrib_init(int argc, char *argv[])
{
#define	 check(ol,al)		if (argv[i][ol] || \
				badarg(argc-i-1,argv+i+1,al)) \
				goto badopt
#define	 check_bool(olen,var)		switch (argv[i][olen]) { \
				case '\0': var = !var; break; \
				case 'y': case 'Y': case 't': case 'T': \
				case '+': case '1': var = 1; break; \
				case 'n': case 'N': case 'f': case 'F': \
				case '-': case '0': var = 0; break; \
				default: goto badopt; }
  char	*curout = NULL;
  char	*prms = NULL;
  char	*binval = NULL;
  int	bincnt = 0;
  int	rval;
  int	i;
  int complete = 0;
  /* global program name */
  progname = argv[0] = fixargv0(argv[0]);
  gargv = argv;
  gargc = argc;
#if defined(_WIN32) || defined(_WIN64)
  _setmaxstdio(2048);		/* increase file limit to maximum */
#endif
  /* initialize calcomp routines early */
  initfunc();
  setcontextcal(RCCONTEXT);

  /* initialize switch variables again in case of reinit */
  header = 1;			/* output header? */
  force_open = 0;			/* truncate existing output? */
  recover = 0;			/* recover previous output? */
  accumulate = 1;			/* input rays per output record */
  contrib = 0;			/* computing contributions? */
  using_stdout = 0;		/* are we using stdout? */
  imm_irrad = 0;			/* compute immediate irradiance? */
  lim_dist = 0;			/* limit distance? */
  outbright = 0; /* output one component brightness */
  report_intvl = 0;		/* reporting interval (seconds) */
  /* option city */
  for (i = 1; i < argc; i++) {
    /* expand arguments */
    while ((rval = expandarg(&argc, &argv, i)) > 0)
      ;
    if (rval < 0) {
      sprintf(errmsg, "cannot expand '%s'", argv[i]);
      error(SYSTEM, errmsg);
    }
    if (argv[i] == NULL || argv[i][0] != '-')
      break;			/* break from options */
    if (!strcmp(argv[i], "-version")) {
      puts(VersionID);
      quit(0);
    }
    if (!strcmp(argv[i], "-defaults") ||
        !strcmp(argv[i], "-help")) {
      override_options();
      rcprintdefaults();
      quit(0);
    }
    rval = getrenderopt(argc-i, argv+i);
    if (rval >= 0) {
      i += rval;
      continue;
    }
    switch (argv[i][1]) {
      case 'n':			/* number of cores */
        check(2,"i");
        nproc = atoi(argv[++i]);
        if (nproc <= 0)
          error(USER, "bad number of processes");
        break;
      case 'V':			/* output contributions */
        check_bool(2,contrib);
        break;
      case 'x':			/* x resolution */
        check(2,"i");
        xres = atoi(argv[++i]);
        break;
      case 'y':			/* y resolution */
        check(2,"i");
        yres = atoi(argv[++i]);
        break;
      case 'w':			/* warnings */
        rval = (erract[WARNING].pf != NULL);
        check_bool(2,rval);
        if (rval) erract[WARNING].pf = wputsrc;
        else erract[WARNING].pf = NULL;
        break;
      case 'e':			/* expression */
        check(2,"s");
        scompile(argv[++i], NULL, 0);
        break;
      case 'l':			/* limit distance */
        if (argv[i][2] != 'd')
          goto badopt;
        check_bool(3,lim_dist);
        break;
      case 'I':			/* immed. irradiance */
        check_bool(2,imm_irrad);
        break;
      case 'f':			/* file or force or format */
        if (!argv[i][2]) {
          check(2,"s");
          loadfunc(argv[++i]);
          break;
        }
        if (argv[i][2] == 'o') {
          check_bool(3,force_open);
          break;
        }
        setformat(argv[i]+2);
        break;
      case 'o':			/* output */
        check(2,"s");
        curout = argv[++i];
        break;
      case 'c':			/* input rays per output */
        check(2,"i");
        accumulate = atoi(argv[++i]);
        break;
      case 'r':			/* recover output */
        check_bool(2,recover);
        break;
      case 'h':			/* header output */
        check_bool(2,header);
        break;
      case 'p':			/* parameter setting(s) */
        check(2,"s");
        set_eparams(prms = argv[++i]);
        break;
      case 'b':			/* bin expression/count */
        if (argv[i][2] == 'n') {
          check(3,"s");
          bincnt = (int)(eval(argv[++i]) + .5);
          break;
        }
        check(2,"s");
        binval = argv[++i];
        break;
      case 'm':			/* modifier name */
        check(2,"s");
        addmodifier(argv[++i], curout, prms, binval, bincnt);
        break;
      case 'M':			/* modifier file */
        check(2,"s");
        addmodfile(argv[++i], curout, prms, binval, bincnt);
        break;
      case 't':			/* reporting interval */
        check(2,"i");
        report_intvl = atoi(argv[++i]);
        break;
      case 'Z':			/* brightness output */
        check_bool(2,outbright);
        break;
      default:
        goto badopt;
    }
  }
  if (nmods <= 0)
    error(USER, "missing required modifier argument");
  /* override some option settings */
  override_options();
  /* initialize object types */
  initotypes();
  /* initialize urand */
  if (rand_samp) {
    srandom((long)time(0));
    initurand(0);
  } else {
    srandom(0L);
    initurand(2048);
  }
  /* set up signal handling */
  sigdie(SIGINT, "Interrupt");
#ifdef SIGHUP
  sigdie(SIGHUP, "Hangup");
#endif
  sigdie(SIGTERM, "Terminate");
#ifdef SIGPIPE
  sigdie(SIGPIPE, "Broken pipe");
#endif
#ifdef SIGALRM
  sigdie(SIGALRM, "Alarm clock");
#endif
#ifdef	SIGXCPU
  sigdie(SIGXCPU, "CPU limit exceeded");
  sigdie(SIGXFSZ, "File size exceeded");
#endif
#ifdef	NICE
  nice(NICE);			/* lower priority */
#endif
  if (i != argc)
    goto badopt;

  complete = 1;
  badopt:
  if (!complete) {
    fprintf(stderr,
            "Usage: %s [-n nprocs][-V][-c count][-r][-e expr][-f source][-o ospec][-p p1=V1,p2=V2][-b binv][-bn N] {-m mod | -M file} [rtrace options] octree\n",
            progname);
    sprintf(errmsg, "command line error at '%s'", argv[i]);
    error(USER, errmsg);
  }
  //ignore all header flags
  header = 0;
  return nproc;

#undef	check
#undef	check_bool
}

void
rcontrib_loadscene(char* ocn) {
  /* get octree */
  octname = ocn;
  readoct(octname, ~(IO_FILES | IO_INFO), &thescene, NULL);
  nsceneobjs = nobjects;

  /* PMAP: set up & load photon maps */
  ray_init_pmap();

  marksources();      /* find and mark sources */

  /* PMAP: init photon map for light source contributions */
  initPmapContrib(&modconttab, nmods);

  setambient();      /* initialize ambient calculation */
}


#ifdef __cplusplus
}
#endif
