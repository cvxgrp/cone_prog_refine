#include "test.h"
#include "cones.h"
#include "mini_cblas.h"
#include "problem.h"




static const char *  test_isin_kexp(double * pi_z){

    mu_assert("Exp cone projection: y < 0", pi_z[1] >= 0);

    if  (pi_z[1] == 0)
        mu_assert("Exp cone projection: y == 0 and not (x <= 0 and z >= 0)", 
            (pi_z[0] <= 0) && (pi_z[2] >= 0));
    else
        mu_assert("Exp cone projection: y > 0 and not y exp (x/y) >= z", 
            pi_z[1] * exp(pi_z[0] / pi_z[1]) <= pi_z[2]);

return 0;
}

static const char *  test_isin_kexp_star(double * x){

    mu_assert("Exp cone projection: u > 0", x[0] <= 0);

    if  (x[0] == 0)
        mu_assert("Exp cone projection: u == 0 and not (v >= 0 and w >= 0)", 
            (x[1] >= 0) && (x[2] >= 0));
    else
        mu_assert("Exp cone projection: u < 0 and not -u e(v/u) <= w ", 
            - x[0] * exp(x[1] / x[0]) <= exp(1.) * x[2]);

return 0;

}

static const char * test_exp_cone_proj(){

    double z[6+1], 
    pi_z[6+1];

    double pi_z_m_z[3];

    int i,j;
    cone_prog_refine_workspace workspace;

    workspace.m = 0;
    workspace.n = 0;
    workspace.size_zero = 0;
    workspace.size_nonneg = 0;
    workspace.num_sec_ord = 0;
    workspace.sizes_sec_ord = NULL;
    workspace.num_exp_pri = 1;
    workspace.num_exp_dua = 1;
    workspace.z = z;
    workspace.pi_z = pi_z;

    for (j=0; j<1000; j++){
        random_uniform_vector(6+1, z, -1, 1, j*12345);
        
        embedded_cone_projection(&workspace);
       
       
      if (DEBUG_PRINT)
        printf("\nTesting exp cone projection\n"); 
        
        if (DEBUG_PRINT){
            printf("testing primal\n");
        

        for (i= 0; i<3;i++){
            printf("z[%d] = %e, pi_z[%d] = %e, (pi_z-z)[%d] = %e\n", 
                i, z[i], i, pi_z[i], i, pi_z[i] - z[i]);
        }}
        
        test_isin_kexp(pi_z);

        pi_z_m_z[0] = pi_z[0] - z[0];
        pi_z_m_z[1] = pi_z[1] - z[1];
        pi_z_m_z[2] = pi_z[2] - z[2];

        if (DEBUG_PRINT) printf("duality gap %e\n", (pi_z[0] * pi_z_m_z[0] +
            pi_z[1] * pi_z_m_z[1] +
            pi_z[2] * pi_z_m_z[2]));

        mu_assert("primal Pi(z)^T(Pi(z) - z) != 0 ",
         (fabs(pi_z[0] * pi_z_m_z[0] +
            pi_z[1] * pi_z_m_z[1] +
            pi_z[2] * pi_z_m_z[2]) < 1E-15));


        test_isin_kexp_star(pi_z_m_z);

       if (DEBUG_PRINT){ 
        printf("testing dual\n");
        for (i= 3; i<6;i++){
            printf("z[%d] = %e, pi_z[%d] = %e, (pi_z-z)[%d] = %e\n", 
                i, z[i], i, pi_z[i], i, pi_z[i] - z[i]);
        }}
        
        test_isin_kexp_star(pi_z+3);

        pi_z_m_z[0] = pi_z[3] + z[3];
        pi_z_m_z[1] = pi_z[4] + z[4];
        pi_z_m_z[2] = pi_z[5] + z[5];

        if (DEBUG_PRINT) printf("duality gap %e\n", 
            (pi_z[3] * pi_z_m_z[0] +
            pi_z[4] * pi_z_m_z[1] +
            pi_z[5] * pi_z_m_z[2]));

        mu_assert("dual Pi(z)^T(Pi(z) - z) != 0 ",
         fabs(pi_z[3] * pi_z_m_z[0] +
            pi_z[4] * pi_z_m_z[1] +
            pi_z[5] * pi_z_m_z[2])<1E-15);

        test_isin_kexp(pi_z_m_z);

     

        }
     return 0;
 }


#define LEN_TEST_SOCP_CONE 10


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
    double z[7];
    int i,j,k;
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

    for (j=0; j<10; j++){
        random_uniform_vector(7, workspace.z, -1, 1,j);
        embedded_cone_projection(&workspace);
       
       if (DEBUG_PRINT){
        printf("\nTesting cone projection\n"); 
        for (i= 0; i<7;i++){
            printf("z[%d] = %f, pi_z[%d] = %f\n", i, 
                workspace.z[i], i, workspace.pi_z[i]);
         }
     }


        for (k = 0; k < lensol + lenzero; k++)
            mu_assert("real cone projection error", 
                workspace.pi_z[k] == workspace.z[k]);
        
        for (k = lensol + lenzero; k < lensol +lenzero + lennonneg; k++)
            mu_assert("non-neg cone projection error", 
                workspace.pi_z[k] >= 0.);
        
        mu_assert("error, pi_z[-1] < 0", workspace.pi_z[7-1] >= 0);
        }
     return 0;
 }

#define EMB_CONE_PROJ_DER_SIZE 23
#define DZ_RANGE 1E-8
#define DPIZ_RANGE 1E-14

static const char * test_embedded_cone_projection_derivative() {

    const int size_sol = 2;
    const int size_zero = 2;
    const int size_nonneg = 2;
    const int num_sec_ord = 1;
    const int len_socp_cones[1] = {10};

    const int size = EMB_CONE_PROJ_DER_SIZE;

    double z[EMB_CONE_PROJ_DER_SIZE], 
    pi_z[EMB_CONE_PROJ_DER_SIZE], 
    dz[EMB_CONE_PROJ_DER_SIZE], 
    z_p_dz[EMB_CONE_PROJ_DER_SIZE], 
    z_m_dz[EMB_CONE_PROJ_DER_SIZE], 
    dpi_z[EMB_CONE_PROJ_DER_SIZE], 
    pi_z_p_dz[EMB_CONE_PROJ_DER_SIZE],
    pi_z_m_dz[EMB_CONE_PROJ_DER_SIZE];

    int i,j, k;
    int equal;
    double normerr;

    cone_prog_refine_workspace workspace;


    initialize_workspace(
        0, 
        size_sol,
        size_zero, /*size of zero cone*/
        size_nonneg, /*size of non-negative cone*/
        num_sec_ord, /*number of second order cones*/
        (const int *) len_socp_cones, /*sizes of second order cones*/
        1, /*number of exponential primal cones*/
        1, /*number of exponential dual cones*/
        NULL, /*pointers to columns of A, in CSC format*/
        NULL, /*indeces of rows of A, in CSC format*/
        NULL, /*elements of A, in CSC format*/
        NULL, /*m-vector*/
        NULL, /*n-vector*/
        z,
        &workspace);

    workspace.pi_z = pi_z;

    for (j=0; j<20; j++){
        random_uniform_vector(size, z, -1, 1, j*1234);
        random_uniform_vector(size, dz, -DZ_RANGE, 
                                DZ_RANGE, j*5678);

        if (DEBUG_PRINT) printf("\nTest #%d of numerical accuracy of DN(z)\n",j);


        /* workspace.pi_z = Pi workspace.z */
        embedded_cone_projection(&workspace);

        for (k = 0; k < 1; k++){

        if (DEBUG_PRINT) printf("\nscaling dz by (0.9)^%d\n",k);

        /*Pi(z + dz) */
        for (i= 0; i<size;i++) z_p_dz[i] = z[i] + dz[i];
        workspace.z = z_p_dz;
        workspace.pi_z = pi_z_p_dz;
        embedded_cone_projection(&workspace);

        /*Pi(z - dz) */
        for (i = 0; i<size;i++) z_m_dz[i] = z[i] - dz[i];
        workspace.z = z_m_dz;
        workspace.pi_z = pi_z_m_dz;
        embedded_cone_projection(&workspace);
        
        /*re-assign correct workspace*/
        workspace.z = z;
        workspace.pi_z = pi_z;


        memset(dpi_z, 0., sizeof(double)*EMB_CONE_PROJ_DER_SIZE);

        embedded_cone_projection_derivative(
            &workspace,
            dz,
            dpi_z,
            1);

        if (DEBUG_PRINT){
        printf("\nTesting cone projection derivative\n");
        for (i= 0; i<size;i++){
           printf("z[%d] = %.2e, pi_z[%d] = %.2e, dz[%d] = %.2e, dpi_z[%d] = %.2e, (pi_z_p_dz - pi_z)[%d] = %.2e\n", 
               i, z[i], i, pi_z[i], i, dz[i], i, dpi_z[i], i, pi_z_p_dz[i] - pi_z[i]);
        }
    }


        equal = 0;
        if (DEBUG_PRINT) printf("\n\n((Pi(z + dz) - Pi(z - dz))/2 - DPi(z)dz)\n");
        if (DEBUG_PRINT) normerr = cblas_dnrm2(size, dz, 1);
        for (i = 0; i <size; i++){
            if (DEBUG_PRINT)

                    printf("[%d] = %e,  \n", i,
                        ((pi_z_p_dz[i] - pi_z_m_dz[i])/2. - dpi_z[i]));

            if (fabs((pi_z_p_dz[i] - pi_z_m_dz[i])/2. - dpi_z[i])>DPIZ_RANGE) {
                equal = -1;} 
        }
        if (equal == 0) break;
        

        for (i= 0; i<size;i++) dz[i] *= (0.9);
}

        mu_assert("error, (Pi(z + dz) - Pi(z - dz))/2 != DPi(z) dz", !equal);

        }
     return 0;
 }
