"""
Copyright 2019 Enzo Busseti, Walaa Moursi, and Stephen Boyd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import numba as nb


@nb.jit(nb.float64[:](nb.int64, nb.int32[:], nb.int32[:], nb.float64[:], nb.float64[:]),
        nopython=True)
def csc_matvec(m, col_pointers, row_indeces, mat_elements, vector):
    """Multiply (m,n) matrix by (n) vector. Matrix is in compressed sparse cols fmt."""
    result = np.zeros(m)
    n = len(col_pointers) - 1
    assert len(vector) == n

    for j in range(n):
        i_s = row_indeces[
            col_pointers[j]:col_pointers[j + 1]]
        elements = mat_elements[
            col_pointers[j]:col_pointers[j + 1]]

        for cur, i in enumerate(i_s):
            result[i] += elements[cur] * vector[j]

    return result


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
