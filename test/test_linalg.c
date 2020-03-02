#include "test.h"
#include "linalg.h"

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
vector,
1
);

if (DEBUG_PRINT) printf("%f, %f, %f\n",result[0], result[1], result[2] );

mu_assert("wrong result CSC matrix vector multiplication",
(result[0] == 2.43) &&
(result[1] == 0.4) &&
(result[2] == 1.56));

csc_matvec(
n, /*number of columns*/
col_pointers, 
row_indeces,
mat_elements,
result,
vector,
0
);

if (DEBUG_PRINT) printf("%f, %f, %f\n",result[0], result[1], result[2]);

mu_assert("wrong result minus CSC matrix vector multiplication",
(fabs(result[0]) < 1E-15) &&
(fabs(result[1]) < 1E-15) &&
(fabs(result[2]) < 1E-15));

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
    vector,
    1
    );

    if (DEBUG_PRINT) printf("%f, %f, %f\n",result[0], result[1], result[2] );

    mu_assert("wrong result CSC matrix vector multiplication",
    (result[0] == 2.61) &&
    (result[1] == 0.4) &&
    (result[2] == 1.5));

    csr_matvec(
    n, /*number of columns*/
    row_pointers, 
    col_indeces,
    mat_elements,
    result,
    vector,
    0
    );

    if (DEBUG_PRINT) printf("%f, %f, %f\n",result[0], result[1], result[2]);

    mu_assert("wrong result minus CSC matrix vector multiplication",
    (fabs(result[0]) < 1E-15) &&
    (fabs(result[1]) < 1E-15) &&
    (fabs(result[2]) < 1E-15));

    return 0;
  }
