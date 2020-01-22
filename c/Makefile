lsqr: src/lsqr.c src/lsqrblas.c
	cc -pedantic -Wall -c -Iinclude src/lsqr.c

cones: src/cones.c
	cc -pedantic -Wall -c -Iinclude -fpic src/cones.c
	gcc -shared -o libcones.so cones.o
