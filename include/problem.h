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
#include <cone_prog_refine.h>

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
    cone_prog_refine_workspace * workspace);

/*
result = result + DN(z) * vector
*/
int normalized_residual_matvec(
    cone_prog_refine_workspace * workspace,
    double * result, 
    double * vector /*It gets changed but then restored.*/
    );

/*
result = result + DN(z)^T * vector
*/
int normalized_residual_vecmat(
    cone_prog_refine_workspace * workspace,
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
