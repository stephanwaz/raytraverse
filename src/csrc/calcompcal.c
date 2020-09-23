
#include  "calcompcal.h"
#include "caldefn.c"



char *
setcontextcal(			/* set a new context path */
	char  *ctx
)
{
    char  *cpp;

    if (ctx == NULL)
	return(context);		/* just asking */
    while (*ctx == CNTXMARK)
	ctx++;				/* skip past marks */
    if (!*ctx) {
	context[0] = '\0';		/* empty means clear context */
	return(context);
    }
    cpp = context;			/* start context with mark */
    *cpp++ = CNTXMARK;
    do {				/* carefully copy new context */
	if (cpp >= context+MAXCNTX)
	    break;			/* just copy what we can */
	if (isid(*ctx))
	    *cpp++ = *ctx++;
	else {
	    *cpp++ = '_'; ctx++;
	}
    } while (*ctx);
    while (cpp[-1] == CNTXMARK)		/* cannot end in context mark */
	cpp--;
    *cpp = '\0';
    return(context);
}
