#include "cone_prog_refine.h"
#include "problem.h"

#include "test.h"

#define TEST_CPR_M 12
#define TEST_CPR_N 9
#define TEST_CPR_SIZE_ZERO 2
#define TEST_CPR_SIZE_NONNEG 5


/*Temporarily defined here because N macro from test.h conflicts.*/
double cblas_dnrm2(const int n, const double *X, const int incX);


static double A_elements_cpr[32] = {
        0.24140296, 0.46029897, 0.8460227 , 0.68658545, 0.58383685,
       0.36225789, 0.52523198, 0.49818503, 0.17863091, 0.40690994,
       0.74128545, 0.926194  , 0.26751764, 0.93814559, 0.95026184,
       0.02635934, 0.6072343 , 0.07554401, 0.37964013, 0.75843112,
       0.04483735, 0.1612756 , 0.17945269, 0.47983984, 0.55213953,
       0.97647449, 0.99311338, 0.91531049, 0.64490518, 0.26995289,
       0.45523676, 0.80031283};
static const int A_col_pointers_cpr[10] = { 0,  5,  6, 11, 12, 17, 22, 25, 30, 32};
static const int A_row_indeces_cpr[32] = {1,  5,  7,  8,  9,  4,  4,  6,  7,  8, 11,  9,  0,  2,  3,  6, 11,
        1,  2,  3,  5,  6,  2,  7,  8,  0,  4,  6,  9, 10,  8,  9};


static const char * test_cone_prog_refine(){

    double z[TEST_CPR_M + TEST_CPR_N + 1];
    double oldnorm, mynorm;
    double b[TEST_CPR_M];
    double c[TEST_CPR_N];
    int k;

    const int sizes_sec_ord_cones[] = {5};

    cone_prog_refine_workspace workspace;

    for (k = 0; k < 10; k++){

    if (DEBUG_PRINT) 
        printf("Cone prog refine test %d\n\n", k);

    random_uniform_vector(TEST_CPR_M + TEST_CPR_N + 1, 
        z, -10, 10, (k+1)*1234);
    random_uniform_vector(TEST_CPR_M, b, 
            -10, 10, (k+1)*123456);
    random_uniform_vector(TEST_CPR_N, c, 
           -10, 10, (k+1)*12345678);

    random_uniform_vector(32, A_elements_cpr, 
           -1, 1, (k+1)*123);

    initialize_workspace(
        TEST_CPR_M, 
        TEST_CPR_N,
        TEST_CPR_SIZE_ZERO, /*size of zero cone*/
        TEST_CPR_SIZE_NONNEG, /*size of non-negative cone*/
        1, /*number of second order cones*/
        sizes_sec_ord_cones, /*sizes of second order cones*/
        0, /*number of exponential primal cones*/
        0, /*number of exponential dual cones*/
        A_col_pointers_cpr, /*pointers to columns of A, in CSC format*/
        A_row_indeces_cpr, /*indeces of rows of A, in CSC format*/
        A_elements_cpr, /*elements of A, in CSC format*/
        b, /*m-vector*/
        c, /*n-vector*/
        z,
        &workspace);

    projection_and_normalized_residual(&workspace);
    oldnorm = cblas_dnrm2(workspace.m + workspace.n+1, 
                          workspace.norm_res_z, 1);


cone_prog_refine(
    TEST_CPR_M, /*number of rows of A*/
    TEST_CPR_N, /*number of columns of A*/
    TEST_CPR_SIZE_ZERO, /*size of zero cone*/
    TEST_CPR_SIZE_NONNEG, /*size of non-negative cone*/
    1, /*number of second order cones*/
    sizes_sec_ord_cones, /*sizes of second order cones*/
    0, /*number of exponential primal cones*/
    0, /*number of exponential dual cones*/
    A_col_pointers_cpr, /*pointers to columns of A, in CSC format*/
    A_row_indeces_cpr, /*indeces of rows of A, in CSC format*/
    A_elements_cpr, /*elements of A, in CSC format*/
    b, /*m-vector*/
    c, /*n-vector*/
    z, /* (m+n+1)-vector, 
        approximate primal-dual embedded solution,
        will be overwritten by refined solution*/
    30, /*number of lsqr iterations*/
    1E-8, /*lambda*/
    2, /*number of refine iterations*/
    DEBUG_PRINT /*print informations on convergence*/
    );

    projection_and_normalized_residual(&workspace);
    mynorm = cblas_dnrm2(workspace.m + workspace.n+1, 
                          workspace.norm_res_z, 1);

    mu_assert("residual norm has not decreased after refinn",
        mynorm < oldnorm);

    }


return 0;}
