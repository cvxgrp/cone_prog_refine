"""
Copyright (C) Enzo Busseti 2017-2019.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
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
