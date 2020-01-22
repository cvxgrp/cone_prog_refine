31 Aug 2007: First version of README.

So far there is no makefile for this version of lsqr.c.
To compile lsqr.c with the include file cblas.h
in the present directory, say this:

   cc -g -c -I. lsqr.c

A stricter compile command is

 cc  -pedantic -Wall -c -I. lsqr.c

Note that // is used for comments.
(cc  -ansi ...  leads to many messages.)


File lsqrblas.c contains the subset of CBLAS used by lsqr.c
(and a few more).  Where possible, optimized or vendor-supplied
CBLAS should be used.
