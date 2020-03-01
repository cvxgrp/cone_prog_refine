#include <stdlib.h>
#include <time.h>

/* Simple Macros from http://www.jera.com/techinfo/jtns/jtn002.html */
#define mu_assert(message, test)  do {if (!(test)) return message; } while (0)
#define mu_run_test(test) do {const char *message = test(); tests_run++;                  \
    if (message) return message;} while (0)

