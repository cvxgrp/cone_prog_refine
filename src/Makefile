CC = gcc
OSX_CBLAS_AND_LAPACK = "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/Headers/"
CFLAGS = -I"../include" -ansi -pedantic -Wall -O3 -I$(OSX_CBLAS_AND_LAPACK)
LFLAGS = -lm -lcblas -llapack

cone_prog_refine : main.o cones.o mini_cblas.o linalg.o problem.o
	cc -o cone_prog_refine main.o cones.o mini_cblas.o $(LFLAGS)

main.o : main.c 
	$(CC) -c main.c $(CFLAGS) $(LIBS)

cones.o : cones.c 
	$(CC) -c cones.c $(CFLAGS) $(LIBS)

problem.o : problem.c 
	$(CC) -c problem.c $(CFLAGS) $(LIBS)

linalg.o : linalg.c 
	$(CC) -c linalg.c $(CFLAGS) $(LIBS)

mini_cblas.o : mini_cblas.c 
	$(CC) -c mini_cblas.c $(CFLAGS) $(LIBS)

test: ../test/test.c ../test/test.h cones.o mini_cblas.o linalg.o problem.o
	$(CC) -o ../test/runtests ../test/test.c cones.o mini_cblas.o linalg.o problem.o $(CFLAGS) $(LIBS) $(LFLAGS); \
	chmod a+x ../test/runtests; ../test/runtests

clean :
	rm cone_prog_refine main.o cones.o mini_cblas.o linalg.o problem.o ../test/runtests
