#ifdef __cplusplus
extern "C" {
#endif

#include "rtrace.c"
#include "rtinit.h"

extern void
rtrace_setup(				/* initialize processes */
	int  nproc
)
{
	long  nextflush = (!vresolu | (hresolu <= 1)) * hresolu;
	if (imm_irrad)
		castonly = 0;
	else if (castonly)
		nproc = 1;		/* don't bother multiprocessing */
	if ((nextflush > 0) & (nproc > nextflush)) {
		error(WARNING, "reducing number of processes to match flush interval");
		nproc = nextflush;
	}
	if (nproc > 1) {		/* start multiprocessing */
		ray_popen(nproc);
		ray_fifo_out = printvals;
	}
}

extern void
rtrace_call(				/* run rtrace process */
        char *fname
)
{
  unsigned long  vcount = (hresolu > 1) ? (unsigned long)hresolu*vresolu
                                        : (unsigned long)vresolu;
  long  nextflush = (!vresolu | (hresolu <= 1)) * hresolu;
  int  something2flush = 0;
  FILE  *fp;
  double	d;
  FVECT  orig, direc;
  /* set up input */
  if (fname == NULL)
    fp = stdin;
  else if ((fp = fopen(fname, "r")) == NULL) {
    sprintf(errmsg, "cannot open input file \"%s\"", fname);
    error(SYSTEM, errmsg);
  }
  if (inform != 'a')
    SET_FILE_BINARY(fp);
  /* set up output */
  setoutput(outvals);
  switch (outform) {
    case 'a': putreal = puta; break;
    case 'f': putreal = putf; break;
    case 'd': putreal = putd; break;
    case 'c':
      if (outvals[0] && (outvals[1] || !strchr("vrx", outvals[0])))
        error(USER, "color format only with -ov, -or, -ox");
      putreal = putrgbe; break;
    default:
      error(CONSISTENCY, "botched output format");
  }
  if (hresolu > 0) {
    if (vresolu > 0)
      fprtresolu(hresolu, vresolu, stdout);
    else
      fflush(stdout);
  }
  /* process file */
  while (getvec(orig, inform, fp) == 0 &&
         getvec(direc, inform, fp) == 0) {

    d = normalize(direc);
    if (d == 0.0) {				/* flush request? */
      if (something2flush) {
        if (ray_pnprocs > 1 && ray_fifo_flush() < 0)
          error(USER, "child(ren) died");
        bogusray();
        fflush(stdout);
        nextflush = (!vresolu | (hresolu <= 1)) * hresolu;
        something2flush = 0;
      } else
        bogusray();
    } else {				/* compute and print */
      rtcompute(orig, direc, lim_dist ? d : 0.0);
      /* flush if time */
      if (!--nextflush) {
        if (ray_pnprocs > 1 && ray_fifo_flush() < 0)
          error(USER, "child(ren) died");
        fflush(stdout);
        nextflush = hresolu;
      } else
        something2flush = 1;
    }
    if (ferror(stdout))
      error(SYSTEM, "write error");
    if (vcount && !--vcount)		/* check for end */
      break;
  }
  if (ray_pnprocs > 1) {				/* clean up children */
    if (ray_fifo_flush() < 0)
      error(USER, "unable to complete processing");
  }
  if (fflush(stdout) < 0)
    error(SYSTEM, "write error");
  if (vcount)
    error(USER, "unexpected EOF on input");
  if (fname != NULL)
    fclose(fp);
}

#ifdef __cplusplus
}
#endif
