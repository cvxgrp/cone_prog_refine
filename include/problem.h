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


/* 
result = result + Q * vector
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
    );

/* 
result = result + Q^T * vector
*/
void Q_vecmat(
    const int m,
    const int n,
    const int * A_col_pointers, 
    const int * A_row_indeces,
    const double * A_data,
    const double * b,
    const double * c,
    double * result,
    const double * vector
    );

/*
N(z) and Pi(z).
*/
void projection_and_normalized_residual(
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
    );

/*
result = result + DN(z) * vector
*/
void normalized_residual_matvec(
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
    const double * pi_z, /*Useful for fast derivatives.*/
    double * result,
    const double * vector
    );

/*
result = result + DN(z)^T * vector
*/
void normalized_residual_vecmat(
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
    const double * pi_z, /*Useful for fast derivatives.*/
    double * result,
    const double * vector
    );
