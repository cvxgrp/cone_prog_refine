#include "test.h"
#include "problem.h"
#include "math.h"
#include <string.h>


static const char * test_Q_matvec(){

/*A = [[0.,   0.,   0.1],
       [0.,   0.20, 0.  ],
       [0.8, 0.,   0.23]]

as CSC matrix */
int i;
const double mat_elements[4] = {0.8, 0.20, 0.1, 0.23};
const int col_pointers[4] = {0, 1, 2, 4};
const int row_indeces[4] = {2, 1, 0, 2};

/*b = [3,5,7] */
const double b[3] = {3,5,7};

/*c = [11,13,17] */
const double c[3] = {11,13,17};

/*Q = [[0, 0, 0,       0.,   0.,   0.8, 11],
       [0, 0, 0,       0.,   0.20, 0. , 13 ],
       [0, 0, 0,       0.1, 0.,   0.23, 17],
       [0.,  0., -0.1,   0, 0, 0,     3],
       [0.,  -0.20, 0.,   0, 0, 0,    5],
       [-0.8, 0.,  -0.23, 0, 0, 0,     7],
       [-11, -13, -17, -3, -5, -7 , 0]
       ]

as CSC matrix */


double result[7] = {0,0,0,0,0,0,0};
const double vector[7] = {1,2,3,4,5,6,7};
const double numpy_result[7] = { 81.8 ,   92.  ,  120.78,   
    20.7 ,   34.6 ,   47.51, -167};


Q_matvec(
    3,
    3,
    col_pointers, 
    row_indeces,
    mat_elements,
    b,
    c,
    result,
    vector,
    1
    );

if (DEBUG_PRINT){
        printf("\nTesting Q_matvec\n"); 
        for (i= 0; i<7;i++){
            printf("result[%d] = %f, numpy_result[%d] = %f\n", 
                i, result[i], i, numpy_result[i]);
         }}

mu_assert("wrong result Q matrix vector multiplication",
(result[0] == numpy_result[0]) &&
(result[1] == numpy_result[1]) &&
(result[2] == numpy_result[2]) &&
(result[3] == numpy_result[3]) &&
(result[4] == numpy_result[4]) &&
(result[5] == numpy_result[5]) &&
(result[6] == numpy_result[6]));

result[0] =0;
result[1]=0;
result[2]=0;
result[3]=0;
result[4]=0;
result[5]=0;
result[6]=0;

Q_matvec(
    3,
    3,
    col_pointers, 
    row_indeces,
    mat_elements,
    b,
    c,
    result,
    vector,
    0
    );

if (DEBUG_PRINT){
        printf("\nTesting Q_matvec with - sign\n"); 
        for (i= 0; i<7;i++){
            printf("result[%d] = %f, numpy_result[%d] = %f\n", 
                i, result[i], i, -numpy_result[i]);
         }}

mu_assert("wrong result Q matrix vector multiplication",
(result[0] == -numpy_result[0]) &&
(result[1] == -numpy_result[1]) &&
(result[2] == -numpy_result[2]) &&
(result[3] == -numpy_result[3]) &&
(result[4] == -numpy_result[4]) &&
(result[5] == -numpy_result[5]) &&
(result[6] == -numpy_result[6]));


return 0;
}

static const char * test_normalized_residual(){

/*A = [[0.,   0.,   0.1],
       [0.,   0.20, 0.  ],
       [0.8, 0.,   0.23]]

as CSC matrix */
int i;
const double mat_elements[4] = {0.8, 0.20, 0.1, 0.23};
const int col_pointers[4] = {0, 1, 2, 4};
const int row_indeces[4] = {2, 1, 0, 2};

/*b = [3,5,7] */
const double b[3] = {3,5,7};

/*c = [11,13,17] */
const double c[3] = {11,13,17};

/*Q = [[0, 0, 0,       0.,   0.,   0.8, 11],
       [0, 0, 0,       0.,   0.20, 0. , 13 ],
       [0, 0, 0,       0.1, 0.,   0.23, 17],
       [0.,  0., -0.1,   0, 0, 0,     3],
       [0.,  -0.20, 0.,   0, 0, 0,    5],
       [-0.8, 0.,  -0.23, 0, 0, 0,     7],
       [-11, -13, -17, -3, -5, -7 , 0]
       ]

as CSC matrix */


double result[7] = {0,0,0,0,0,0,0};
double pi_z[7] = {0,0,0,0,0,0,0};
const double vector[7] = {1,2,3,4,-5,6,1};
const double numpy_result[7] = {15.8 ,   13.  ,   18.78,    2.7 ,   -0.4 ,    5.51, -142. };


projection_and_normalized_residual(
    3,
    3,
    1,
    2,
    0,
    NULL,
    0,
    0,
    col_pointers, 
    row_indeces,
    mat_elements,
    b,
    c,
    result,
    pi_z,
    vector
    );



if (DEBUG_PRINT){
        printf("\nTesting Q_matvec\n"); 
        for (i= 0; i<7;i++){
            printf("result[%d] = %f, pi_z[%d] = %f\n", 
                i, result[i], i, pi_z[i]);
         }}

mu_assert("wrong result Q matrix vector multiplication",
(fabs(result[0] - numpy_result[0]) < 1E-15) &&
(fabs(result[1] - numpy_result[1]) < 1E-15) &&
(fabs(result[2] - numpy_result[2]) < 1E-15) &&
(fabs(result[3] - numpy_result[3]) < 1E-15) &&
(fabs(result[4] - numpy_result[4]) < 1E-15) &&
(fabs(result[5] - numpy_result[5]) < 1E-15) &&
(fabs(result[6] - numpy_result[6]) < 1E-15));

return 0;


}


static const char * test_normalized_residual_matvec(){

    int i;
    int k;
    const int m = 3;
    const int n = 3;
    const int size_zero = 1;
    const int size_nonneg = 2;
    const int num_sec_ord = 0;
    const int * sizes_sec_ord = NULL;
    const int num_exp_pri = 0;
    const int num_exp_dua = 0;


    const double A_data[4] = {0.8, 0.20, 0.1, 0.23};
    const int A_col_pointers[4] = {0, 1, 2, 4};
    const int A_row_indeces[4] = {2, 1, 0, 2};
    const double b[3] = {3,5,7};
    const double c[3] = {11,13,17};
    double z[7];
    double pi_z[7];
    double norm_res_z[7];
    double result[7]= {0., 0., 0., 0., 0., 0., 0.};
    double d_pi_z[7];
    double dz[7]; 
    double z_p_dz[7];
    double check[7];

    double error;

    for (k = 0; k < NUM_CONES_TESTS; k++){

        if (DEBUG_PRINT) printf("\nTesting DN(z) * x\n");

    random_uniform_vector(n+m+1, z, 
                        -1, 1, (1+k)*1234);

    /* Setting z[n+m] = 1. or -1 simplifies the test.*/ 
    /* z[n+m] = 1.; */ 

    projection_and_normalized_residual(
    m,
    n,
    size_zero,
    size_nonneg,
    num_sec_ord,
    sizes_sec_ord,
    num_exp_pri,
    num_exp_dua,
    A_col_pointers, 
    A_row_indeces,
    A_data,
    b,
    c,
    norm_res_z,
    pi_z,
    (const double *) z
    );

    random_uniform_vector(n+m+1, dz, 
                        -1E-8, 1E-8, (1+k)*5678);

    for (i = 0; i < n+m+1; i++) {
        z_p_dz[i] = z[i] + dz[i];
                /*   printf("z[%d] = %e, dz[%d] = %e\n", 
                i, z[i],i, dz[i]); */
                }

    result[0] = 0.;
    result[1] = 0.;
    result[2] = 0.;
    result[3] = 0.;
    result[4] = 0.;
    result[5] = 0.;
    result[6] = 0.;

    normalized_residual_matvec(
        m,
        n,
        size_zero,
        size_nonneg,
        num_sec_ord,
        sizes_sec_ord,
        num_exp_pri,
        num_exp_dua,
        A_col_pointers, 
        A_row_indeces,
        A_data,
        b,
        c,
        (const double *)  z,
        (const double *) pi_z, /*Used by cone derivatives.*/
        (const double *) norm_res_z, /*Used by second term of derivative*/
        result,
        d_pi_z, /*Used internally.*/
        dz /*It gets changed.*/
        );


        projection_and_normalized_residual(
        m,
        n,
        size_zero,
        size_nonneg,
        num_sec_ord,
        sizes_sec_ord,
        num_exp_pri,
        num_exp_dua,
        A_col_pointers, 
        A_row_indeces,
        A_data,
        b,
        c,
        check, /*N(z + dz)*/
        pi_z, 
        (const double *) z_p_dz 
        );

        for (i = 0; i < n+m+1; i++){
            error = check[i] - norm_res_z[i] - result[i];
            if (DEBUG_PRINT) printf("(N(z + dz) - N(z) - DN(z) dz)[%d] = %e\n", 
                i, error);
            mu_assert("Large error normalized_residual_matvec",(fabs(error) < 1E-12));

            if (DEBUG_PRINT)  printf("(N(z + dz)[%d] = %e, N(z)[%d] = %e,  DN(z)dz[%d] = %e\n", 
                i, check[i],i, norm_res_z[i],i, result[i]);

        }

    }





    return 0;
}

static const char * test_Q_vecmat(){

    int i;
    int k;
    const int m = 3;
    const int n = 3;

    const double A_data[4] = {0.8, 0.20, 0.1, 0.23};
    const int A_col_pointers[4] = {0, 1, 2, 4};
    const int A_row_indeces[4] = {2, 1, 0, 2};
    const double b[3] = {3,5,7};
    const double c[3] = {11,13,17};

    /*We build the matrix by using matvec and vecmat,
    and check that they are equal if transposed.*/
    int j;
    double result_matvec [7][7];
    double result_vecmat [7][7];
    double d [7];

    for (k = 0; k < 10; k++){

        if (DEBUG_PRINT)  printf("\nTesting Q^T\n");

    /* Setting z[n+m] = 1. or -1 simplifies the test.*/ 
    /* z[n+m] = 1.; */

    for (j = 0; j < 7; j++){
        memset(result_matvec[j], 0, sizeof(double) * (7));
        memset(d, 0, sizeof(double) * (7));
        d[j] = 1.;

    Q_matvec(
    m,
    n,
    A_col_pointers, 
    A_row_indeces,
    A_data,
    b,
    c,
    result_matvec[j],
    d,
    1
    );

        for (i = 0; i < 7; i++){
            if (DEBUG_PRINT)  printf("%.2e  ", result_matvec[j][i]);
        }
        if (DEBUG_PRINT)  printf("\n");

        }


    for (j = 0; j < 7; j++){
        memset(result_vecmat[j], 0, sizeof(double) * (7));
        memset(d, 0, sizeof(double) * (7));

        d[j] = 1.;

    Q_matvec(
    m,
    n,
    A_col_pointers, 
    A_row_indeces,
    A_data,
    b,
    c,
    result_vecmat[j],
    d,
    0
    );

        }

    if (DEBUG_PRINT) printf("\n\n\n");
    for (j = 0; j < 7; j++){
    for (i = 0; i < 7; i++){
        if (DEBUG_PRINT)  printf("%.2e  ", result_vecmat[i][j]);
        
        mu_assert("Q transpose not equal to Q",
            (result_matvec[i][j] == result_vecmat[j][i]));

    }
    if (DEBUG_PRINT) printf("\n");
    }

    }

    return 0;
}


static const char * test_normalized_residual_vecmat(){

    int i;
    int k;
    const int m = 3;
    const int n = 3;
    const int size_zero = 1;
    const int size_nonneg = 2;
    const int num_sec_ord = 0;
    const int * sizes_sec_ord = NULL;
    const int num_exp_pri = 0;
    const int num_exp_dua = 0;


    const double A_data[4] = {0.8, 0.20, 0.1, 0.23};
    const int A_col_pointers[4] = {0, 1, 2, 4};
    const int A_row_indeces[4] = {2, 1, 0, 2};
    const double b[3] = {3,5,7};
    const double c[3] = {11,13,17};
    double z[7];
    double pi_z[7];
    double norm_res_z[7];
    double internal[7];
    double internal2[7];
    double d[7]; 


    /*We build the matrix by using matvec and vecmat,
    and check that they are equal if transposed.*/
    int j;
    double result_matvec [7][7];
    double result_vecmat [7][7];

    for (k = 0; k < 10; k++){

        if (DEBUG_PRINT)  printf("\nTesting DN(z)^T\n");

    random_uniform_vector(n+m+1, z, 
                        -1, 1, (1+k)*1234);

    /* Setting z[n+m] = 1. or -1 simplifies the test.*/ 
    /* z[n+m] = 1.; */

    if (DEBUG_PRINT)  printf("z[n+m] = %f\n", z[n+m]);

    projection_and_normalized_residual(
    m,
    n,
    size_zero,
    size_nonneg,
    num_sec_ord,
    sizes_sec_ord,
    num_exp_pri,
    num_exp_dua,
    A_col_pointers, 
    A_row_indeces,
    A_data,
    b,
    c,
    norm_res_z,
    pi_z,
    (const double *) z
    );


    for (j = 0; j < 7; j++){
        memset(result_matvec[j], 0, sizeof(double) * (7));
        memset(d, 0, sizeof(double) * (7));
        memset(internal, 0, sizeof(double) * (7));
        d[j] = 1.;

    normalized_residual_matvec(
        m,
        n,
        size_zero,
        size_nonneg,
        num_sec_ord,
        sizes_sec_ord,
        num_exp_pri,
        num_exp_dua,
        A_col_pointers, 
        A_row_indeces,
        (const double *)A_data,
        (const double *)b,
        (const double *)c,
        (const double *)  z,
        (const double *) pi_z, /*Used by cone derivatives.*/
        (const double *) norm_res_z, /*Used by second term of derivative*/
        result_matvec[j],
        internal, /*Used internally.*/
        d /*It gets changed.*/
        );

        for (i = 0; i < 7; i++){
            if (DEBUG_PRINT)  printf("%.2e  ", result_matvec[j][i]);
        }
        if (DEBUG_PRINT)  printf("\n");

        }


    for (j = 0; j < 7; j++){
        memset(result_vecmat[j], 0, sizeof(double) * (7));
        memset(d, 0, sizeof(double) * (7));
        memset(internal, 0, sizeof(double) * (7));
        memset(internal2, 0, sizeof(double) * (7));

        d[j] = 1.;

    normalized_residual_vecmat(
        m,
        n,
        size_zero,
        size_nonneg,
        num_sec_ord,
        sizes_sec_ord,
        num_exp_pri,
        num_exp_dua,
        A_col_pointers, 
        A_row_indeces,
        (const double *)A_data,
        (const double *)b,
        (const double *)c,
        (const double *) z,
        (const double *) pi_z, /*Used by cone derivatives.*/
        (const double *) norm_res_z, /*Used by second term of derivative*/
        result_vecmat[j],
        internal, /*Used internally.*/
        internal2, /*Used internally.*/
        d /*It gets changed.*/
        );

        }

        if (DEBUG_PRINT)  printf("\n\n\n");
        for (j = 0; j < 7; j++){
        for (i = 0; i < 7; i++){
            if (DEBUG_PRINT)  printf("%.2e  ", result_vecmat[i][j]);
            
            mu_assert("DN(z) transpose not equal to DN(z)",
                (result_matvec[i][j] == result_vecmat[j][i]));

        }
        if (DEBUG_PRINT)  printf("\n");
    }

     if (DEBUG_PRINT) {
                printf("\nError * 1E8:\n");
        for (j = 0; j < 7; j++){
        for (i = 0; i < 7; i++){
            printf("%.2e   ", (result_vecmat[i][j] - result_matvec[j][i])*1E8);

        }
        printf("\n");
    }}

    }

    return 0;
}

