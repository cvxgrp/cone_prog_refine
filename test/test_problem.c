#include "test.h"
#include "problem.h"

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
    vector
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

return 0;
}


