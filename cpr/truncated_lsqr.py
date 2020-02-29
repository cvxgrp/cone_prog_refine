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

# Implementation of LSQR following the algorithm description
# in "LSQR: An Algorithm for Sparse Linear Equations and Sparse
# Least Squares", by C. Paige and M. Saunders, 1982.

# The only termination criteria used is the maximum number
# of iterations max_iter.

import numpy as np


def d2norm(a, b):
    """np.sqrt(a**2 + b**2) that limits overflow"""
    scale = np.abs(a) + np.abs(b)
    if scale == 0.:
        return 0.
    scaled_a = a / scale
    scaled_b = b / scale
    return scale * np.sqrt(scaled_a**2 + scaled_b**2)

# @nb.jit(nopython=True)
# def _sym_ortho(a, b):

#     # c, s, rho = _sym_ortho(rhobar, beta)
#     """
#     Stable implementation of Givens rotation.

#     Notes
#     -----
#     The routine 'SymOrtho' was added for numerical stability. This is
#     recommended by S.-C. Choi in [1]_.  It removes the unpleasant potential of
#     ``1/eps`` in some important places (see, for example text following
#     "Compute the next plane rotation Qk" in minres.py).

#     References
#     ----------
#     .. [1] S.-C. Choi, "Iterative Methods for Singular Linear Equations
#            and Least-Squares Problems", Dissertation,
#            http://www.stanford.edu/group/SOL/dissertations/sou-cheng-choi-thesis.pdf

#     """
#     if b == 0:
#         return np.sign(a), 0, np.abs(a)
#     elif a == 0:
#         return 0, np.sign(b), np.abs(b)
#     elif np.abs(b) > np.abs(a):
#         tau = a / b
#         s = np.sign(b) / np.sqrt(1 + tau * tau)
#         c = s * tau
#         r = b / s
#     else:
#         tau = b / a
#         c = np.sign(a) / np.sqrt(1 + tau * tau)
#         s = c * tau
#         r = a / c
#     return c, s, r


#     The matrix A is intended to be large and sparse.  It is accessed
#     by means of subroutine calls of the form
#
#                call aprod ( mode, m, n, x, y, leniw, lenrw, iw, rw )
#
#     which must perform the following functions:
#
#                If mode = 1, compute  y = y + A*x.
#                If mode = 2, compute  x = x + A(transpose)*y.
#
#     The vectors x and y are input parameters in both cases.
#     If  mode = 1,  y should be altered without changing x.
#     If  mode = 2,  x should be altered without changing y.
#     The parameters leniw, lenrw, iw, rw may be used for workspace
#     as described below.


def aprod(mode, m, n, x, y, leniw, lenrw, iw, rw):

    pass


def truncated_lsqr(m, n, matvec, rmatvec, b, max_iter=30):

    x = np.zeros(n)

    beta = np.linalg.norm(b)
    if beta == 0:
        return x
    u = b / beta

    v = rmatvec(u)
    alpha = np.linalg.norm(v)
    if alpha == 0:
        return x
    v /= alpha

    w = np.copy(v)

    phi_bar = float(beta)
    rho_bar = float(alpha)

    for i in range(max_iter):

        # print(i)

        # continue the bidiagonalization
        u = matvec(v) - alpha * u
        beta = np.linalg.norm(u)
        if beta == 0:
            return x
        u /= beta

        v = rmatvec(u) - beta * v
        alpha = np.linalg.norm(v)
        if alpha == 0:
            return x
        v /= alpha

        # costruct and apply next orthogonal transformation

        rho = d2norm(rho_bar, beta)

        #c, s, rho = _sym_ortho(rho_bar, beta)

        # rho = np.sqrt(rho_bar**2 + beta**2)
        c = rho_bar / rho
        s = beta / rho

        theta = s * alpha
        rho_bar = -c * alpha
        phi = c * phi_bar
        phi_bar = s * phi_bar

        # update x and w
        x = x + (phi / rho) * w
        w = v - (theta / rho) * w

    return x
