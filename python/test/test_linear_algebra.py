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


import unittest
import numpy as np

import scipy.sparse as sp

from cpr import *
import scipy.sparse as sp

import time


class LSQRTest(unittest.TestCase):

    def test_dense(self):

        for m, n in [(30, 20), (50, 10), (10, 50), (30, 30), (100, 100)]:
            print(m, n)

            A = np.random.randn(m, n)
            # if m == n:
            #     print('unscaled cond num', np.linalg.cond(A))
            #     A /= np.linalg.norm(A, axis=0)
            #     A = A.T / np.linalg.norm(A, axis=1)
            # if m == n:
            #     print('unscaled cond num', np.linalg.cond(A))
            #     d = 1. / np.sqrt(np.abs(np.diag(A)))
            #     A = np.diag(d) @ A @ np.diag(d)
            print('cond num', np.linalg.cond(A))
            x = np.random.randn(n)
            b = A @ x

            matvec = lambda x: A @ x
            rmatvec = lambda y: A.T @ y

            # print(x)

            my_x = truncated_lsqr(m, n, matvec,
                                  rmatvec, b, max_iter=3 * max(m, n))

            # print(my_x)
            print(b - A@my_x)

            self.assertTrue(np.allclose(b, A@my_x))

    def test_sparse(self):

        for m, n in [(30, 20), (50, 10), (10, 50), (100, 100)]:
            print(m, n)
            A = sp.random(m, n).tocsc()
            x = np.random.randn(n)
            b = A @ x

            matvec = lambda x: A @ x
            rmatvec = lambda y: A.T @ y

            # print(x)

            my_x = truncated_lsqr(m, n, matvec,
                                  rmatvec, b, max_iter=max(m, n))

            # print(my_x)
            print(b - A@my_x)

            self.assertTrue(np.allclose(b, A@my_x))


class SparseLinalgTest(unittest.TestCase):

    def test_CSC(self):

        m, n = 40, 30
        A_csc = sp.random(m, n, density=.2, format='csc')

        b = np.random.randn(n)
        self.assertTrue(np.allclose(csc_matvec(A_csc.shape[0],
                                               A_csc.indptr,
                                               A_csc.indices,
                                               A_csc.data, b),
                                    A_csc @ b))

    def test_CSR(self):

        m, n = 40, 30
        A_csc = sp.random(m, n, density=.2, format='csr')

        b = np.random.randn(n)
        self.assertTrue(np.allclose(csr_matvec(A_csc.indptr,
                                               A_csc.indices,
                                               A_csc.data, b),
                                    A_csc @ b))
