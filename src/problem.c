/*
*  Cone Program Refinement
*
*  Copyright (C) 2020 Enzo Busseti
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

#include "problem.h"
#include "linalg.h"
#include "mini_cblas.h"

/* 
result = result + Q * vector
Q = 
[[0    A^T  c]
 [-A   0    b]
 [-c^T -b^T 0]]]

A is (m x n) in CSC format.
Q has shape (m+n+1, m+n+1).
*/
void Q_matvec(
    const int m,
    const int n,
    const int * A_col_pointers, 
    const int * A_row_indeces,
    const double * A_data,
    const double * b,
    const double * c,
    double * result,
    const double * vector
    ){

    /*result[0:n] = result[0:n] + A^T * vector[n:n+m]*/
    csr_matvec(
    n, /*number of rows of A^T*/
    A_col_pointers, 
    A_row_indeces,
    A_data,
    result,
    vector + n,
    1
    );

    /*result[0:n] = result[0:n] + c * vector[n+m]*/
    cblas_daxpy(n, vector[m+n], c, 1, result, 1);

    /*result[n:n+m] = result[n:n+m] - A * vector[0:n]*/
    csc_matvec(
    n, /*number of columns of A*/
    A_col_pointers, 
    A_row_indeces,
    A_data,
    result + n ,
    vector,
    0
    );

    /*result[n:n+m] = result[n:n+m] + b * vector[n+m]*/
    cblas_daxpy(m, vector[m+n], b, 1, result + n, 1);

    /*result[n+m] -= c^T * vector[:n]*/
    result[n+m] -= cblas_ddot(n, c, 1, vector, 1);

    /*result[n+m] -= b^T * vector[n:m]*/
    result[n+m] -= cblas_ddot(m, b, 1, vector + n, 1);

}
