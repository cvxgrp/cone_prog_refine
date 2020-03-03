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
#include <string.h>

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
    const bool forward
    ){

    /*result[0:n] = result[0:n] (+-) A^T * vector[n:n+m]*/
    csr_matvec(
    n, /*number of rows of A^T*/
    A_col_pointers, 
    A_row_indeces,
    A_data,
    result,
    vector + n,
    forward
    );

    /*result[0:n] = result[0:n] + c * vector[n+m]*/
    cblas_daxpy(n, forward ? vector[m+n]:-vector[m+n], 
        c, 1, result, 1);

    /*result[n:n+m] = result[n:n+m] - A * vector[0:n]*/
    csc_matvec(
    n, /*number of columns of A*/
    A_col_pointers, 
    A_row_indeces,
    A_data,
    result + n ,
    vector,
    !forward
    );

    /*result[n:n+m] = result[n:n+m] + b * vector[n+m]*/
    cblas_daxpy(m, forward ? vector[m+n]:-vector[m+n], 
        b, 1, result + n, 1);

    /*result[n+m] -= c^T * vector[:n]*/
    if (forward) result[n+m] -= cblas_ddot(n, c, 1, vector, 1);
    else result[n+m] += cblas_ddot(n, c, 1, vector, 1);

    /*result[n+m] -= b^T * vector[n:m]*/
    if (forward) result[n+m] -= cblas_ddot(m, b, 1, vector + n, 1);
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

    /*result = 0.*/
    memset(result, 0, sizeof(double) * (m+n+1));

    /*result += Q * pi_z */
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

/*
result = result + DN(z) * vector
*/
int normalized_residual_matvec(
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
    const double * z,
    const double * pi_z, /*Used by cone derivatives.*/
    const double * norm_res_z, /*Used by second term of derivative*/
    double * result,
    double * d_pi_z, /*Used as internal storage space.*/
    double * vector /*It gets changed.*/
    ){

    int non_diff = 0;

    if (fabs(z[n+m]) == 0.) return -1;

    /* vector /= |w| */
    cblas_dscal(n+m+1, 1./fabs(z[n+m]), vector, 1);

    /* result += vector */
    cblas_daxpy(n+m+1, 1, (const double *)vector, 1, result, 1);

    /* d_pi_z = DPi(z) * vector */
    non_diff = embedded_cone_projection_derivative(
    (const double *)z, 
    (const double *)pi_z,
    (const double *)vector,
    d_pi_z,
    n,
    size_zero, 
    size_nonneg
    /*const vecsize num_second_order,
    const vecsize * sizes_second_order
    const vecsize num_exp_pri,
    const vecsize num_exp_dua*/
    );

    /* result -= d_pi_z */
    cblas_daxpy(n+m+1, -1, (const double *)d_pi_z, 1, result, 1);


    /* result += Q d_pi_z; */
    Q_matvec(
        m,
        n,
        A_col_pointers, 
        A_row_indeces,
        A_data,
        b,
        c,
        result,
        (const double *) d_pi_z,
        1
        );
    
    /*result += (vector[n+m] * -sign(w)) * N(z) */
    cblas_daxpy(n+m+1, ((const double *)vector)[n+m] * (z[n+m] > 0 ? -1. : 1.), 
        (const double *) norm_res_z, 1, result, 1);

    /* vector *= |w| */
    cblas_dscal(n+m+1, fabs(z[n+m]), vector, 1);

    return non_diff;
}

/*
result = result + DN(z)^T * vector
*/
int normalized_residual_vecmat(
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
    const double * z,
    const double * pi_z, /*Used by cone derivatives.*/
    const double * norm_res_z, /*Used by second term of derivative*/
    double * result,
    double * internal, /*Used as internal storage space.*/
    double * internal2, 
    /*Used as internal storage space, change DPi(x) so that it adds to result and remove this.*/
    double * vector /*It gets changed but then restored.*/
    ){

    int non_diff = 0;
    
    if (fabs(z[n+m]) == 0.) return -1;

    /* vector /= |w| */
    cblas_dscal(n+m+1, 1./fabs(z[n+m]), vector, 1);

    /* result += vector */
    cblas_daxpy(n+m+1, 1., (const double *)vector, 1, result, 1);

    /* result[n+m] += (vector^T N(z)) * (-sign(z[n+m])) */
    if (z[n+m] > 0)
        result[n+m] -= cblas_ddot(n+m+1, (const double *)vector, 1, norm_res_z, 1);
    else 
        result[n+m] += cblas_ddot(n+m+1, (const double *)vector, 1, norm_res_z, 1);


    /*internal = -vector */
    cblas_dcopy(m+n+1, (const double *)vector, 1, internal, 1);
    cblas_dscal(m+n+1, -1., internal, 1);

    /* internal += Q^T vector */
    Q_matvec(
        m,
        n,
        A_col_pointers, 
        A_row_indeces,
        A_data,
        b,
        c,
        internal,
        (const double *) vector,
        0 /*Q^T */
        );

    /* internal2 = DPi(z)^T * internal . TODO add transpose to embedded_cone_projection_derivative */
    non_diff = embedded_cone_projection_derivative(
    (const double *)z, 
    (const double *)pi_z,
    (const double *)internal,
    internal2,
    n,
    size_zero, 
    size_nonneg
    /*const vecsize num_second_order,
    const vecsize * sizes_second_order
    const vecsize num_exp_pri,
    const vecsize num_exp_dua*/
    );

    /* result += internal2 */
    cblas_daxpy(n+m+1, 1, (const double *)internal2, 1, result, 1);

    /*scale back vector*/
    cblas_dscal(n+m+1, fabs(z[n+m]), vector, 1);

    return non_diff;


}


void normalized_residual_aprod(
    const int mode, const int lsqr_m, const int lsqr_n, 
    double * x, double * y, void *UsrWrk){

    /* y = y + A*x */
    if (mode == 1){

    normalized_residual_matvec(
    *((struct lsqr_workspace *)UsrWrk)->m,
    *((struct lsqr_workspace *)UsrWrk)->n,
    *((struct lsqr_workspace *)UsrWrk)-> size_zero,
    *((struct lsqr_workspace *)UsrWrk)->size_nonneg,
    *((struct lsqr_workspace *)UsrWrk)->num_sec_ord,
    *((struct lsqr_workspace *)UsrWrk)->sizes_sec_ord,
    *((struct lsqr_workspace *)UsrWrk)->num_exp_pri,
    *((struct lsqr_workspace *)UsrWrk)->num_exp_dua,
    *((struct lsqr_workspace *)UsrWrk)->A_col_pointers, 
    *((struct lsqr_workspace *)UsrWrk)->A_row_indeces,
    *((struct lsqr_workspace *)UsrWrk)->A_data,
    *((struct lsqr_workspace *)UsrWrk)->b,
    *((struct lsqr_workspace *)UsrWrk)->c,
    ((struct lsqr_workspace *)UsrWrk)->z,
    ((struct lsqr_workspace *)UsrWrk)->pi_z, /*Used by cone derivatives.*/
    ((struct lsqr_workspace *)UsrWrk)->norm_res_z, /*Used by second term of derivative*/
    y,
    ((struct lsqr_workspace *)UsrWrk)->internal, /*Used internally.*/
    x /*It gets changed but then restored.*/
    );
    }

    /* x = x + A(transpose)*y */
    if (mode == 2){

    normalized_residual_vecmat(
    *((struct lsqr_workspace *)UsrWrk)->m,
    *((struct lsqr_workspace *)UsrWrk)->n,
    *((struct lsqr_workspace *)UsrWrk)-> size_zero,
    *((struct lsqr_workspace *)UsrWrk)->size_nonneg,
    *((struct lsqr_workspace *)UsrWrk)->num_sec_ord,
    *((struct lsqr_workspace *)UsrWrk)->sizes_sec_ord,
    *((struct lsqr_workspace *)UsrWrk)->num_exp_pri,
    *((struct lsqr_workspace *)UsrWrk)->num_exp_dua,
    *((struct lsqr_workspace *)UsrWrk)->A_col_pointers, 
    *((struct lsqr_workspace *)UsrWrk)->A_row_indeces,
    *((struct lsqr_workspace *)UsrWrk)->A_data,
    *((struct lsqr_workspace *)UsrWrk)->b,
    *((struct lsqr_workspace *)UsrWrk)->c,
    ((struct lsqr_workspace *)UsrWrk)->z,
    ((struct lsqr_workspace *)UsrWrk)->pi_z, /*Used by cone derivatives.*/
    ((struct lsqr_workspace *)UsrWrk)->norm_res_z, /*Used by second term of derivative*/
    x,
    ((struct lsqr_workspace *)UsrWrk)->internal, /*Used as internal storage space.*/
    ((struct lsqr_workspace *)UsrWrk)->internal2, 
    /*Used as internal storage space, change DPi(x) so that it adds to result and remove this.*/
    y /*It gets changed but then restored.*/
    );


    }

}



