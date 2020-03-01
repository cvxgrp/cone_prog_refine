#include <stdlib.h>
#include <time.h>

/* Simple Macros from http://www.jera.com/techinfo/jtns/jtn002.html */
#define mu_assert(message, test)  do {if (!(test)) return message; } while (0)
#define mu_run_test(test) do {const char *message = test(); tests_run++;                  \
    if (message) return message;} while (0)

/* Write random doubles to array. */
static int random_uniform_vector(int len_array, double *array, 
                                double low, double high, unsigned int seed)
{
    int i;
    /*srand((unsigned int)time(NULL));*/
    srand(seed);
    if (high <= low){return -1;};
    for (i=0;i<len_array;i++)
        array[i] = ((double)rand()/(double)(RAND_MAX)) * (high - low) + low;
    return 0;
}
