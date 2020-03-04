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
#ifndef CONE_PROG_REFINE_H
#define CONE_PROG_REFINE_H

#define MAX_CONE_PROG_REFINE_BACKTRACKS 10

int cone_prog_refine_alloc(
    const int m, 
    const int n,
    double ** norm_res,
    double ** pi_z,
    double ** internal,  /*Used by DN(z)*/
    double ** internal2,  /*Used by DN(z)*/
    double ** u, /*Used by LSQR*/
    double ** v, /*Used by LSQR*/
    double ** w, /*Used by LSQR*/
    double ** delta /*Used by LSQR*/
    );

int cone_prog_refine(
    const int m, /*number of rows of A*/
    const int n, /*number of columns of A*/
    const int size_zero, /*size of zero cone*/
    const int size_nonneg, /*size of non-negative cone*/
    const int num_sec_ord, /*number of second order cones*/
    const int *sizes_sec_ord, /*sizes of second order cones*/
    const int num_exp_pri, /*number of exponential primal cones*/
    const int num_exp_dua, /*number of exponential dual cones*/
    const int * A_col_pointers, /*pointers to columns of A, in CSC format*/
    const int * A_row_indeces, /*indeces of rows of A, in CSC format*/
    const double * A_data, /*elements of A, in CSC format*/
    const double * b, /*m-vector*/
    const double * c, /*n-vector*/
    double * z, /* (m+n+1)-vector, 
                    approximate primal-dual embedded solution,
                    will be overwritten by refined solution*/
    const int num_lsqr_iters, /*number of lsqr iterations*/
    const int num_iters, /*number of refine iterations*/
    const int print_info /*print informations on convergence*/
    );

#endif
