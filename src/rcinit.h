/* RCSid $Id: func.h,v 2.10 2015/05/20 12:58:31 greg Exp $ */
/*
 * Header file for modifiers using function files.
 *
 * Include after ray.h
 */
#ifndef _RAD_RTINIT_H_
#define _RAD_RTINIT_H_

#ifdef __cplusplus
extern "C" {
#endif

static void rcprintdefaults(void);
extern int rcontrib_init(int  argc, char  *argv[]);
extern void rcontrib_call(char *fname);
extern void rcontrib_clear(void);
static FILE* rcinit2(char *fname);
extern void parental_loop2(char *fname);
extern void feeder_loop2(char *fname);
extern int getvecfp(FVECT vec, FILE *fp);

#ifdef __cplusplus
}
#endif
#endif /* _RAD_RTMAIN_H_ */

