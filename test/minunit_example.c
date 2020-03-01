 /* file minunit_example.c */
 
#include <stdio.h>
#include "minunit.h"
#include <stdlib.h>

 
 int tests_run = 0;
 
 int foo = 7;
 int bar = 4;

 #include <stdio.h>

static int random_uniform_vector(int len_array, double *array, 
                                double low, double high, 
                                unsigned int seed)
{
    srand(seed);
    if (high <= low){return -1;};
    for (int i=0;i<len_array;i++)
        array[i] = ((float)rand()/(float)(RAND_MAX)) * (high - low) + low;
    return 0;
}

 
 static const char * test_foo() {
    double z[20], pi_z[20];
     random_uniform_vector(20, z, 
                            -1,1, 
                            0);

    //const double constz[20] = z[20];




    embedded_cone_projection(
    z, 
    pi_z,
    5,
    5, 
    9
    /*const vecsize num_second_order,
    const vecsize * sizes_second_order
    const vecsize num_exp_pri,
    const vecsize num_exp_dua*/
    );
    for (int i= 0; i<20;i++){
        printf("%f\n", pi_z[i]);
     }

     mu_assert("error, foo != 7", foo == 7);
     return 0;
 }
 
 static const char * test_bar() {
     double a[20];
     random_uniform_vector(20, a, 
                                -1,1, 
                                0);

     for (int i= 0; i<20;i++){
        printf("%f\n", a[i]);
     }


     mu_assert("error, bar != 5", bar == 4);
     return 0;
 }
 
 static const char * all_tests() {
     mu_run_test(test_foo);
     mu_run_test(test_bar);
     return 0;
 }
 
 int main(int argc, char **argv) {
     const char *result = all_tests();
     if (result != 0) {
         printf("%s\n", result);
     }
     else {
         printf("ALL TESTS PASSED\n");
     }
     printf("Tests run: %d\n", tests_run);
 
     return result != 0;
 }