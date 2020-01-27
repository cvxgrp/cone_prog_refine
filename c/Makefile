LOCAL_CBLAS_AND_LAPACK = "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/System/Library/Frameworks/Accelerate.framework/Versions/A/Frameworks/vecLib.framework/Versions/A/Headers/"

lsqr: src/lsqr.c 
	cc -ansi -pedantic -Wall -c -Iinclude -I$(LOCAL_CBLAS_AND_LAPACK) src/lsqr.c

cones: src/cones.c
	cc -ansi -pedantic -Wall -c -Iinclude -I$(LOCAL_CBLAS_AND_LAPACK) -fpic src/cones.c
	gcc -shared -o libcones.so cones.o -lcblas -llapack
