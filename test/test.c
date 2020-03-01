 /* file minunit_example.c */
 
#include <stdio.h>
#include "test.h"
#include <stdlib.h>
#include "cones.h"
#include "linalg.h"

#include <sys/time.h>

 
int tests_run = 0;

#define LENSOL  2
#define LENZERO 2
#define LENNONEG 2
#define N (LENSOL + LENZERO + LENNONEG + 1)
#define DEBUG_PRINT 0
#define NUM_CONES_TESTS 10
#define NUM_BACKTRACKS 10

/* Write random doubles to array. */
static int random_uniform_vector(int len_array, double *array, 
                                double low, double high, unsigned int seed)
{
    int i;
    srand(seed*123456789);
    /*srand(((unsigned int)time(NULL))*1234);*/

    if (high <= low){return -1;};
    for (i=0;i<len_array;i++)
        array[i] = ((double)rand()/(double)(RAND_MAX)) * (high - low) + low;
    return 0;
}

static const char * test_embedded_cone_projection(){

    int lensol = LENSOL;
    int lenzero = LENZERO;
    int lennonneg = LENNONEG;
    
    double z[N], pi_z[N];
    int i,j,k;

    for (j=0; j<NUM_CONES_TESTS; j++){
        random_uniform_vector(N, z, -1, 1,j);
        embedded_cone_projection(z, pi_z, lensol, lenzero, lennonneg);
       
       if (DEBUG_PRINT){
        printf("\nTesting cone projection\n"); 
        for (i= 0; i<N;i++){
            printf("z[%d] = %f, pi_z[%d] = %f\n", i, z[i], i, pi_z[i]);
         }
     }


        for (k = 0; k < lensol + lenzero; k++)
            mu_assert("real cone projection error", pi_z[k] == z[k]);
        
        for (k = lensol + lenzero; k < lensol +lenzero + lennonneg; k++)
            mu_assert("non-neg cone projection error", pi_z[k] >= 0.);
        
        mu_assert("error, pi_z[-1] < 0", pi_z[N-1] >= 0);
        }
     return 0;
 }

/*
static const char * test_embedded_cone_projection(){
const char * result = 0;

result = test_embedded_cone_projection_variable_size(0,0,0);
if (!result) {return result;};
result = test_embedded_cone_projection_variable_size(1,0,0);
if (!result) {return result;};
result = test_embedded_cone_projection_variable_size(0,1,0);
if (!result) {return result;};
result = test_embedded_cone_projection_variable_size(0,0,1);
if (!result) {return result;};
result = test_embedded_cone_projection_variable_size(1,1,0);
if (!result) {return result;};
result = test_embedded_cone_projection_variable_size(0,1,1);
if (!result) {return result;};
result = test_embedded_cone_projection_variable_size(1,0,1);
if (!result) {return result;};
result = test_embedded_cone_projection_variable_size(1,1,1);

return 0;
}
*/


 static const char * test_embedded_cone_projection_derivative() {
    double z[N], pi_z[N], dz[N], z_p_dz[N], dpi_z[N], pi_z_p_dz[N];
    int i,j, k;
    int equal;


    for (j=0; j<NUM_CONES_TESTS; j++){
        random_uniform_vector(N, z, -1, 1, j*1234);
        random_uniform_vector(N, dz, -1, 1, j*5678);
        embedded_cone_projection(z, pi_z, LENSOL, LENZERO, LENNONEG);

        for (k = 0; k < NUM_BACKTRACKS; k++){

        if (DEBUG_PRINT) printf("\nscaling dz by (1/2)^%d\n",k);

        for (i= 0; i<N;i++) z_p_dz[i] = z[i] + dz[i];

        embedded_cone_projection(z_p_dz, pi_z_p_dz, LENSOL, LENZERO, LENNONEG);

        embedded_cone_projection_derivative(z, pi_z, dz, dpi_z, LENSOL, LENZERO, LENNONEG);

        if (DEBUG_PRINT){
        printf("\nTesting cone projection derivative\n");
        for (i= 0; i<N;i++){
           printf("z[%d] = %.2f, pi_z[%d] = %.2f, dz[%d] = %.2f, dpi_z[%d] = %.2f, pi_z_p_dz[%d] = %.2f\n", 
               i, z[i], i, pi_z[i], i, dz[i], i, dpi_z[i], i, pi_z_p_dz[i]);
        }
    }

        /* pi_z + dpi_z == pi_z_p_dz*/
        equal = 0;
        for (i = 0; i <N; i++){
            if (pi_z[i] + dpi_z[i] != pi_z_p_dz[i]) {
                equal = -1;
                break;} 
        }
        if (equal == 0) break;
        

        for (i= 0; i<N;i++) dz[i] /= 2.;
}

        mu_assert("error, pi_z + dpi_z != pi_z_p_dz", !equal);

        }
     return 0;
 }

  static const char * test_csc_matvec(){
    /*

    [[0.,   0.,   0.81],
     [0.,   0.20, 0.  ],
     [0.87, 0.,   0.23]]

    as CSC matrix */
    double mat_elements[4] = {0.87, 0.20, 0.81, 0.23};
    int col_pointers[4] = {0, 1, 2, 4};
    int row_indeces[4] = {2, 1, 0, 2};

    int n = 3;
    
    double vector[3] = {1., 2., 3.};
    double result[3] = {0., 0., 0.};

    csc_matvec(
    n, /*number of columns*/
    col_pointers, 
    row_indeces,
    mat_elements,
    result,
    vector
    );

    if (DEBUG_PRINT) printf("%f, %f, %f\n",result[0], result[1], result[2] );

    mu_assert("wrong result CSC matrix vector multiplication",
    (result[0] == 2.43) &&
    (result[1] == 0.4) &&
    (result[2] == 1.56));

    return 0;
  }
 
static const char * test_csr_matvec(){
    /*

    [[0.,   0.,   0.87],
     [0.,   0.20, 0.  ],
     [0.81, 0.,   0.23]]

    as CSR matrix */
    double mat_elements[4] = {0.87, 0.20, 0.81, 0.23};
    int row_pointers[4] = {0, 1, 2, 4};
    int col_indeces[4] = {2, 1, 0, 2};

    int n = 3;
    
    double vector[3] = {1., 2., 3.};
    double result[3] = {0., 0., 0.};

    csr_matvec(
    n, /*number of columns*/
    row_pointers, 
    col_indeces,
    mat_elements,
    result,
    vector
    );

    if (DEBUG_PRINT) printf("%f, %f, %f\n",result[0], result[1], result[2] );

    mu_assert("wrong result CSC matrix vector multiplication",
    (result[0] == 2.61) &&
    (result[1] == 0.4) &&
    (result[2] == 1.5));

    return 0;
  }

 
 static const char * all_tests() {
     mu_run_test(test_embedded_cone_projection);
     mu_run_test(test_embedded_cone_projection_derivative);
     mu_run_test(test_csc_matvec);
     mu_run_test(test_csr_matvec);
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
