#ifndef LINALG_H
#define LINALG_H

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
#include <stdbool.h>

/*result = result + (sign_plus) * A * vector - (!sign_plus) * A * vector
A in CSC sparse format.*/
void csc_matvec(
    const int n, /*number of columns*/
    const int * col_pointers, 
    const int * row_indeces,
    const double * mat_elements,
    double * result,
    const double * vector,
    const bool sign_plus
    );

/*result = result + A * vector
A in CSR sparse format.*/
void csr_matvec(
    const int m, /*number of rows*/
    const int * row_pointers, 
    const int * col_indeces,
    const double * mat_elements,
    double * result,
    const double * vector,
    const bool sign_plus
    );

#endif
