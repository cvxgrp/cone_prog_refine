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

#include "linalg.h"

/*Multiply (m,n) matrix in compressed sparse 
columns format by n-vector.*/
void csc_matvec(
    const int n, /*number of columns*/
    const int * col_pointers, 
    const int * row_indeces,
    const double * mat_elements,
    double * result,
    const double * vector
    ){
    int j, i;
    for (j = 0; j<n; j++)
        for (i = col_pointers[j]; i < col_pointers[j + 1]; i++)
            result[row_indeces[i]] +=  mat_elements[i] * vector[j];
}

/*Multiply (m,n) matrix in compressed sparse 
rows format by n-vector.*/
void csr_matvec(
    const int m, /*number of rows*/
    const int * row_pointers, 
    const int * col_indeces,
    const double * mat_elements,
    double * result,
    const double * vector
    ){
    int j, i;
    for (i = 0; i<m; i++)
        for (j = row_pointers[i]; j < row_pointers[i + 1]; j++)
            result[i] +=  mat_elements[j] * vector[col_indeces[j]];

        // for i in range(m):
        //  js = col_indeces[row_pointers[i]:row_pointers[i + 1]]
        //  elements = mat_elements[row_pointers[i]:row_pointers[i + 1]]
        //  for cur, j in enumerate(js):
        //     result[i] += vector[j] * elements[cur]



}

/*

@nb.jit(nb.float64[:](nb.int32[:], nb.int32[:], nb.float64[:], nb.float64[:]),
        nopython=True)
def csr_matvec(row_pointers, col_indeces, mat_elements, vector):
    """Multiply (m,n) matrix by (n) vector. Matrix is in compressed sparse rows fmt."""
    m = len(row_pointers) - 1
    result = np.zeros(m)

    for i in range(m):
        js = col_indeces[row_pointers[i]:row_pointers[i + 1]]
        elements = mat_elements[row_pointers[i]:row_pointers[i + 1]]
        for cur, j in enumerate(js):
            result[i] += vector[j] * elements[cur]

    return result
*/
