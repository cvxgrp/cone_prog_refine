#include "test.h"
#include "cones.h"
#include "mini_cblas.h"
#include "problem.h"



#define LEN_TEST_SOCP_CONE 10
#define NUM_CONES_TESTS 10
#define NUM_BACKTRACKS 10

static const char * test_second_order_cone(){

    double z[LEN_TEST_SOCP_CONE+1], 
    pi_z[LEN_TEST_SOCP_CONE+1];

    vecsize len_socp_cones[1] = {LEN_TEST_SOCP_CONE};
    int i,j;
    double normx;
    cone_prog_refine_workspace workspace;

    workspace.m = 0;
    workspace.n = 0;
    workspace.size_zero = 0;
    workspace.size_nonneg = 0;
    workspace.num_sec_ord = 1;
    workspace.sizes_sec_ord = (const int *)len_socp_cones;
    workspace.num_exp_pri = 0;
    workspace.num_exp_dua = 0;
    workspace.z = z;
    workspace.pi_z = pi_z;

    for (j=0; j<30; j++){
        random_uniform_vector(LEN_TEST_SOCP_CONE+1, z, -1, 1,j);

        if (j == 0){
            z[0] = -20;
        }


        if (j == 1){
            z[0] = 20;
        }
        
        embedded_cone_projection(&workspace);
           /* z, pi_z, 0, 0, 0, 1, len_socp_cones); */
       
       if (DEBUG_PRINT){
        printf("\nTesting SOCP cone projection\n"); 
        for (i= 0; i<LEN_TEST_SOCP_CONE+1;i++){
            printf("z[%d] = %f, pi_z[%d] = %f\n", i, z[i], i, pi_z[i]);
         }
     }

     normx = cblas_dnrm2(LEN_TEST_SOCP_CONE-1,pi_z+1,1);

     if (DEBUG_PRINT) printf("||x|| = %e\n", normx);

     mu_assert("second order cone projection error", 
        (pi_z[0] - normx) >= -1E-15);

     /*Also test that (pi_z - z) is in the second order cone */

     cblas_daxpy(LEN_TEST_SOCP_CONE, -1., z, 1, pi_z, 1);

     if (DEBUG_PRINT) printf("dual cone\n");

        if (DEBUG_PRINT)
            for (i= 0; i<LEN_TEST_SOCP_CONE+1;i++){
            printf("(pi_z - z)[%d] = %f\n",  i, pi_z[i]);
         }


     normx = cblas_dnrm2(LEN_TEST_SOCP_CONE-1,pi_z+1,1);

     if (DEBUG_PRINT) printf("||x|| = %e\n", normx);

     mu_assert("second order cone projection error", 
        (pi_z[0] - normx) >= -1E-15);

        }
     return 0;
 }


static const char * test_embedded_cone_projection(){

    int lensol = 2;
    int lenzero = 2;
    int lennonneg = 2;
    double z[7], pi_z[7];
    int i,j,k;
    cone_prog_refine_workspace workspace;


    workspace.m = 0;
    workspace.n = 2;
    workspace.size_zero = 2;
    workspace.size_nonneg = 2;
    workspace.num_sec_ord = 0;
    workspace.sizes_sec_ord = NULL;
    workspace.num_exp_pri = 0;
    workspace.num_exp_dua = 0;
    workspace.z = z;
    workspace.pi_z = pi_z;


    for (j=0; j<NUM_CONES_TESTS; j++){
        random_uniform_vector(7, z, -1, 1,j);
        embedded_cone_projection(
            &workspace);
       
       if (DEBUG_PRINT){
        printf("\nTesting cone projection\n"); 
        for (i= 0; i<7;i++){
            printf("z[%d] = %f, pi_z[%d] = %f\n", i, z[i], i, pi_z[i]);
         }
     }


        for (k = 0; k < lensol + lenzero; k++)
            mu_assert("real cone projection error", pi_z[k] == z[k]);
        
        for (k = lensol + lenzero; k < lensol +lenzero + lennonneg; k++)
            mu_assert("non-neg cone projection error", pi_z[k] >= 0.);
        
        mu_assert("error, pi_z[-1] < 0", pi_z[7-1] >= 0);
        }
     return 0;
 }


 static const char * test_embedded_cone_projection_derivative() {
    double z[7], 
    pi_z[7], 
    dz[7], 
    z_p_dz[7], 
    dpi_z[7], 
    pi_z_p_dz[7];
    int i,j, k;
    int equal;

    cone_prog_refine_workspace workspace;


    initialize_workspace(
        0, 
        2,
        2, /*size of zero cone*/
        2, /*size of non-negative cone*/
        0, /*number of second order cones*/
        NULL, /*sizes of second order cones*/
        0, /*number of exponential primal cones*/
        0, /*number of exponential dual cones*/
        NULL, /*pointers to columns of A, in CSC format*/
        NULL, /*indeces of rows of A, in CSC format*/
        NULL, /*elements of A, in CSC format*/
        NULL, /*m-vector*/
        NULL, /*n-vector*/
        z,
        &workspace);

    workspace.pi_z = pi_z;

    for (j=0; j<NUM_CONES_TESTS; j++){
        random_uniform_vector(7, z, -1, 1, j*1234);
        random_uniform_vector(7, dz, -1E-8, 1E-8, j*5678);


        /* workspace.pi_z = Pi workspace.z */
        embedded_cone_projection(&workspace);

        for (k = 0; k < NUM_BACKTRACKS; k++){

        if (DEBUG_PRINT) printf("\nscaling dz by (0.9)^%d\n",k);

        for (i= 0; i<7;i++) z_p_dz[i] = z[i] + dz[i];

        workspace.z = z_p_dz;
        workspace.pi_z = pi_z_p_dz;

        embedded_cone_projection(&workspace);
        workspace.z = z;
        workspace.pi_z = pi_z;


        embedded_cone_projection_derivative(
            &workspace,
            dz,
            dpi_z);

        if (DEBUG_PRINT){
        printf("\nTesting cone projection derivative\n");
        for (i= 0; i<7;i++){
           printf("z[%d] = %.2f, pi_z[%d] = %.2f, dz[%d] = %.2f, dpi_z[%d] = %.2f, pi_z_p_dz[%d] = %.2f\n", 
               i, z[i], i, pi_z[i], i, dz[i], i, dpi_z[i], i, pi_z_p_dz[i]);
        }
    }

        /* pi_z + dpi_z == pi_z_p_dz*/
        equal = 0;
        for (i = 0; i <7; i++){
            if (DEBUG_PRINT)
                    printf("error[%d] = %e\n", i,
                        pi_z[i] + dpi_z[i] - pi_z_p_dz[i]);
            if (fabs(pi_z[i] + dpi_z[i] - pi_z_p_dz[i])>1E-15) {
                equal = -1;
                break;} 
        }
        if (equal == 0) break;
        

        for (i= 0; i<7;i++) dz[i] *= (0.9);
}

        mu_assert("error, pi_z + dpi_z != pi_z_p_dz", !equal);

        }
     return 0;
 }
