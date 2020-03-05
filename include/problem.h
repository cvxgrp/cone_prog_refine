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
#ifndef PROBLEM_H
#define PROBLEM_H

#include <stdbool.h>


typedef struct {
    int m;
    int n;
    int size_zero;
    int size_nonneg;
    int num_sec_ord;
    const int * sizes_sec_ord;
    int num_exp_pri;
    int num_exp_dua;
    const int * A_col_pointers;
    const int * A_row_indeces;
    const double * A_data;
    const double * b;
    const double * c;
    double * z;
    double * pi_z; /*Used by cone derivatives.*/
    double * norm_res_z; /*Used by second term of derivative*/
    double * internal; /* (n+m+1) array for internal storage space.*/
    double * internal2; /* (n+m+1) array for internal storage space.*/
} lsqr_workspace;


/* 
result = result + (forward) * Q * vector + (!forward) * Q^T * vector
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
    );



/*
N(z) and Pi(z).
*/
int projection_and_normalized_residual(
    lsqr_workspace * workspace);
    /*
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
    );*/

/*
result = result + DN(z) * vector
*/
int normalized_residual_matvec(
    lsqr_workspace * workspace,
    double * result, 
    double * vector /*It gets changed but then restored.*/
    );

/*
result = result + DN(z)^T * vector
*/
int normalized_residual_vecmat(
    lsqr_workspace * workspace,
    double * result, 
    double * vector /*It gets changed but then restored.*/
    );


/*
*   Function used by LSQR
*
*   If mode = 1, compute  y = y + DN * x
*   If mode = 2, compute  x = x + DN^T * y
*
*   lsqr_m = lsqr_n = m+n+1
* 
*/
void normalized_residual_aprod(
    int mode, int lsqr_m, int lsqr_n, 
    double * x, double * y, 
    void * workspace);


#endif
