#include "test.h"
#include "problem.h"
#include "math.h"

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

