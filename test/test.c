 /* file minunit_example.c */
 
#include <stdio.h>
#include "test.h"
#include <stdlib.h>
#include "cones.h"

 
int tests_run = 0;
int foo = 7;
int bar = 4;

 
static const char * test_embedded_cone_projection() {
    
    #define LENSOL  1
    #define LENZERO 1
    #define LENNONEG 1

    #define N (LENSOL + LENZERO + LENNONEG + 1)
    
    double z[N], pi_z[N];
    int i,j;

    for (j=0; j<10; j++){
    
    random_uniform_vector(N, z, -1, 1,j);
    embedded_cone_projection(z, pi_z, LENSOL, LENZERO, LENNONEG);

    for (i= 0; i<N;i++){
        printf("z[%d] = %f, pi_z[%d] = %f\n", i, z[i], i, pi_z[i]);
     }

    mu_assert("error, pi_z[0] != z[0]", pi_z[0] == z[0]);
    mu_assert("error, pi_z[LENSOL] != z[LENSOL]", pi_z[LENSOL] == z[LENSOL]);
    mu_assert("error, pi_z[LENSOL+LENZERO] < 0", pi_z[LENSOL+LENZERO] >= 0);
    mu_assert("error, pi_z[-1] < 0", pi_z[N-1] >= 0);
}
     return 0;
 }
 
 static const char * test_embedded_cone_projection_derivative() {
     mu_assert("error, bar != 5", bar == 4);
     return 0;
 }
 
 static const char * all_tests() {
     mu_run_test(test_embedded_cone_projection);
     mu_run_test(test_embedded_cone_projection_derivative);
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
