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
#include "cones.h"
#include <math.h>

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
    const double * vector,
    const bool sign_plus
    ){

    /*result[0:n] = result[0:n] (+-) A^T * vector[n:n+m]*/
    csr_matvec(
    n, /*number of rows of A^T*/
    A_col_pointers, 
    A_row_indeces,
    A_data,
    result,
    vector + n,
    sign_plus
    );

    /*result[0:n] = result[0:n] + c * vector[n+m]*/
    cblas_daxpy(n, sign_plus ? vector[m+n]:-vector[m+n], 
        c, 1, result, 1);

    /*result[n:n+m] = result[n:n+m] - A * vector[0:n]*/
    csc_matvec(
    n, /*number of columns of A*/
    A_col_pointers, 
    A_row_indeces,
    A_data,
    result + n ,
    vector,
    !sign_plus
    );

    /*result[n:n+m] = result[n:n+m] + b * vector[n+m]*/
    cblas_daxpy(m, sign_plus ? vector[m+n]:-vector[m+n], 
        b, 1, result + n, 1);

    /*result[n+m] -= c^T * vector[:n]*/
    if (sign_plus) result[n+m] -= cblas_ddot(n, c, 1, vector, 1);
    else result[n+m] += cblas_ddot(n, c, 1, vector, 1);

    /*result[n+m] -= b^T * vector[n:m]*/
    if (sign_plus) result[n+m] -= cblas_ddot(m, b, 1, vector + n, 1);
    else result[n+m] += cblas_ddot(m, b, 1, vector + n, 1);

}

/*
N(z) and Pi(z).
*/
int projection_and_normalized_residual(
    const int m,
    const int n,
    const int size_zero,
    const int size_nonneg,
    const int num_sec_ord,
    const int *sizes_sec_ord,
    const int num_exp_pri,
    const int num_exp_dua,
    const int * A_col_pointers, 
    const int * A_row_indeces,
    const double * A_data,
    const double * b,
    const double * c,
    double * result,
    double * pi_z,
    const double * z
    ){

    if (z[n+m] == 0.) return -1;

    /*pi_z = Pi(z)*/
    embedded_cone_projection(
        z, 
        pi_z,
        n,
        size_zero, 
        size_nonneg);

    /*result = Q * pi_z */
    Q_matvec(
    m,
    n,
    A_col_pointers, 
    A_row_indeces,
    A_data,
    b,
    c,
    result,
    pi_z,
    1
    );

    /*result -= pi_z*/
    cblas_daxpy(n+m+1, -1, pi_z, 1, result, 1);

    /*result += z*/
    cblas_daxpy(n+m+1, 1, z, 1, result, 1);

    /*result /= |z[n+m]| */
    cblas_dscal(n+m+1, 1./fabs(z[n+m]), result, 1);

    return 0;

}
