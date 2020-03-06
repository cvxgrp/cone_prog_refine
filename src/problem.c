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
    cone_prog_refine_workspace * workspace){

    int size;

    size = workspace->n + workspace->m + 1;

    if (workspace->z[size-1] == 0.) return -1;

    /*workspace->pi_z = workspace->Pi(z)*/
    embedded_cone_projection(workspace);

    /*N(z) = 0.*/
    memset(workspace->norm_res_z, 0, sizeof(double) * (size));

    /*N(z) += Q * pi_z */
    Q_matvec(
    workspace->m,
    workspace->n,
    workspace->A_col_pointers, 
    workspace->A_row_indeces,
    workspace->A_data,
    workspace->b,
    workspace->c,
    workspace->norm_res_z,
    workspace->pi_z,
    1
    );

    /*N(z) -= pi_z*/
    cblas_daxpy(size, -1, workspace->pi_z, 1, workspace->norm_res_z, 1);

    /*N(z) += z*/
    cblas_daxpy(size, 1, workspace->z, 1, workspace->norm_res_z, 1);

    /*N(z) /= |z[n+m]| */
    cblas_dscal(size, 1./fabs(workspace->z[size-1]), workspace->norm_res_z, 1);

    return 0;

}

/*
result = result + DN(z) * vector
*/
int normalized_residual_matvec(
    cone_prog_refine_workspace * workspace,
    double * result,
    double * vector /*It gets changed.*/
    ){

    int non_diff = 0;
    int size;

    size = workspace->n + workspace->m + 1;

    if (fabs(workspace->z[size-1]) == 0.) return -1;

    /* vector /= |w| */
    cblas_dscal(size, 1./fabs(workspace->z[size-1]), vector, 1);

    /* result += vector */
    cblas_daxpy(size, 1, (const double *)vector, 1, result, 1);


    /* internal = DPi(z) * vector */
    memset(workspace->internal, 0., sizeof(double)*size);
    non_diff = embedded_cone_projection_derivative(
        workspace,
        vector,
        workspace->internal,
        1);

    /* result -= internal */
    cblas_daxpy(size, -1, (const double *)workspace->internal, 1, result, 1);


    /* result += Q internal; */
    Q_matvec(
        workspace->m,
        workspace->n,
        workspace->A_col_pointers, 
        workspace->A_row_indeces,
        workspace->A_data,
        workspace->b,
        workspace->c,
        result,
        (const double *) workspace->internal,
        1
        );
    
    /*result += (vector[n+m] * -sign(w)) * N(z) */
    cblas_daxpy(size, ((const double *)vector)[size-1] * (workspace->z[size-1] > 0 ? -1. : 1.), 
        (const double *) workspace->norm_res_z, 1, result, 1);

    /* vector *= |w| */
    cblas_dscal(size, fabs(workspace->z[size-1]), vector, 1);

    return non_diff;
}

/*
result = result + DN(z)^T * vector
*/
int normalized_residual_vecmat(
    cone_prog_refine_workspace * workspace,
    double * result, 
    double * vector /*It gets changed but then restored.*/
    )
{

    int non_diff = 0;
    int size;
    size = workspace->n + workspace->m + 1;
    
    if (fabs(workspace->z[size-1]) == 0.) return -1;

    /* vector /= |w| */
    cblas_dscal(size, 1./fabs(workspace->z[size-1]), vector, 1);

    /* result += vector */
    cblas_daxpy(size, 1., (const double *)vector, 1, result, 1);

    /* result[n+m] += (vector^T N(z)) * (-sign(z[n+m])) */
    if (workspace->z[size-1] > 0)
        result[size-1] -= cblas_ddot(size, (const double *)vector, 1, workspace->norm_res_z, 1);
    else 
        result[size-1] += cblas_ddot(size, (const double *)vector, 1, workspace->norm_res_z, 1);


    /*internal = -vector */
    cblas_dcopy(size, (const double *)vector, 1, workspace->internal, 1);
    cblas_dscal(size, -1., workspace->internal, 1);

    /* internal += Q^T vector */
    Q_matvec(
        workspace->m,
        workspace->n,
        workspace->A_col_pointers, 
        workspace->A_row_indeces,
        workspace->A_data,
        workspace->b,
        workspace->c,
        workspace->internal,
        (const double *) vector,
        0 /*Q^T */
        );

    /* result += DPi(z)^T * internal . TODO add transpose to embedded_cone_projection_derivative */
    memset(workspace->internal2, 0., sizeof(double)*size);
    non_diff = embedded_cone_projection_derivative(
        workspace,
        (const double *) workspace->internal,
        workspace->internal2,
        2);

    /* result += internal2 */
    cblas_daxpy(size, 1, (const double *)workspace->internal2, 1, result, 1);

    /*scale back vector*/
    cblas_dscal(size, fabs(workspace->z[size-1]), vector, 1);

    return non_diff;


}


void normalized_residual_aprod(
    int mode, int lsqr_m, int lsqr_n, 
    double * x, double * y, 
    void * workspace){

    /*struct cone_prog_refine_workspace * workspace = UsrWrk;*/

    /* y = y + A*x */
    if (mode == 1){

    normalized_residual_matvec(
    (cone_prog_refine_workspace *)workspace,
    y,
    x /*It gets changed but then restored.*/
    );
    }

    /* x = x + A(transpose)*y */
    if (mode == 2){

        normalized_residual_vecmat(
    (cone_prog_refine_workspace *)workspace,
    x, 
    y /*It gets changed but then restored.*/
    );

    }

}



