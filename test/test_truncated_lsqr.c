#include "truncated_lsqr.h"
#include "test.h"
#include "linalg.h"

#define TEST_TRUCATED_LSQR_M 8
#define TEST_TRUCATED_LSQR_N 9

static const double A_elements[21] = {
        0.1694194 , 0.88665261, 0.0562221 , 0.23697255, 0.30359439,
       0.61760815, 0.72203815, 0.80486929, 0.16139777, 0.68472571,
       0.99363244, 0.54555176, 0.07660275, 0.37773523, 0.57609738,
       0.31489492, 0.41576018, 0.99150653, 0.05383216, 0.9596463 ,
       0.40253118};
static const int A_col_pointers[10] = {0,  3,  5,  7,  9, 13, 16, 17, 20, 21};
static const int A_row_indeces[21] = {0, 3, 6, 2, 3, 5, 7, 1, 5, 0, 1, 
                                3, 4, 3, 5, 6, 3, 0, 3, 6, 0};


static void aprod(const int mode, const int m, const int n, 
                  double * x, double * y, void *UsrWrk){


    /* y = y + A*x */
    if (mode == 1){

    csc_matvec(
    TEST_TRUCATED_LSQR_N,
    A_col_pointers, 
    A_row_indeces,
    A_elements,
    y,
    x,
    1
    );


    }

    /* x = x + A(transpose)*y */
    if (mode == 2){

    csr_matvec(
    TEST_TRUCATED_LSQR_N, /*number of rows of A^T*/
    A_col_pointers, 
    A_row_indeces,
    A_elements,
    x,
    y,
    1
    );

    }

}

static const char * test_truncated_lsqr(){


    double real_x[TEST_TRUCATED_LSQR_N];
    double x[TEST_TRUCATED_LSQR_N];
    double v[TEST_TRUCATED_LSQR_N];
    double w[TEST_TRUCATED_LSQR_N];
    double b[TEST_TRUCATED_LSQR_M];
    double lsqr_b[TEST_TRUCATED_LSQR_M];
    double u[TEST_TRUCATED_LSQR_M];
    int i,k;


    for (k=0; k<10;k++){

        printf("\n\nTesting LSQR\n\n");

            memset(b, 0., sizeof(double)*TEST_TRUCATED_LSQR_M);
    memset(lsqr_b, 0., sizeof(double)*TEST_TRUCATED_LSQR_M);

    random_uniform_vector(TEST_TRUCATED_LSQR_N, real_x, 
                -10, 10, k*1234);

    for (i =0; i<TEST_TRUCATED_LSQR_N; i++) printf("real_x[%d] =%f\n", i, real_x[i]);

    /* b = b + A*x */
    csc_matvec(
        TEST_TRUCATED_LSQR_N,
        A_col_pointers, 
        A_row_indeces,
        A_elements,
        b,
        real_x,
        1
        );

    for (i =0; i<TEST_TRUCATED_LSQR_M; i++) printf("b[%d] =%f\n", i, b[i]);

    truncated_lsqr(TEST_TRUCATED_LSQR_M, 
    TEST_TRUCATED_LSQR_N,
    aprod,
    b, /*m-vector*/
    15,
    x, /*result n-vector*/
    u, /*internal m-vector*/
    v, /*internal n-vector*/
    w, /*internal n-vector*/
    NULL /*workspace for aprod*/
    );

    for (i =0; i<TEST_TRUCATED_LSQR_N; i++) {
        printf("(LSQR x)[%d]  - x[%d] = %e\n", i,i, x[i] - real_x[i]);

        /*
        uncomment this if you change shape of matrix
        mu_assert("LSQR solution different than original vector",
        (x[i] == real_x[i]));
        */

    }

    csc_matvec(
        TEST_TRUCATED_LSQR_N,
        A_col_pointers, 
        A_row_indeces,
        A_elements,
        lsqr_b,
        x,
        1
        );

    for (i =0; i<TEST_TRUCATED_LSQR_M; i++) {
        /*printf("LSQR: b[%d] =%f\n", i, lsqr_b[i]);*/
        printf("(LSQR implied b)[%d] - b[%d]: %e\n", i, i, lsqr_b[i] - b[i]);


        mu_assert("LSQR implied b different than b",
            fabs(b[i] - lsqr_b[i]) < 1E-14);


    }
}




return 0;

}
