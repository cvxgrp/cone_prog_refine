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

import time
import numpy as np
from numpy.linalg import norm
import scipy.sparse as sp
import numba as nb

# from scipy.sparse.linalg import lsqr, LinearOperator
# from .lsqr import lsqr

import scipy.sparse as sp

from .cones import prod_cone
from .utils import *
from .sparse_matvec import *


# csc_matvec(A_csc.shape[0], A_csc.indptr, A_csc.indices, A_csc.data, b)

#@nb.jit(nopython=True)


#@nb.jit(nb.types.Tuple((nb.float64[:], nb.float64[:]))(
#    nb.float64[:], nb.float64[:], nb.float64[:],
#    nb.optional(nb.float64), nb.optional(nb.float64)), nopython=True)
@nb.jit(nopython=True)
def xsy2uv(x, s, y, tau=1., kappa=0.):
    n = len(x)
    m = len(s)
    u = np.empty(m + n + 1)
    v = np.empty_like(u)
    u[:n] = x
    u[n:-1] = y
    u[-1] = tau
    v[:n] = 0
    v[n:-1] = s
    v[-1] = kappa
    return u, v


#@nb.jit(nopython=True)
@nb.jit(nb.float64[:](
    nb.float64[:], nb.float64[:], nb.float64[:],
    nb.optional(nb.float64), nb.optional(nb.float64)),
    nopython=True)
def xsy2z(x, s, y, tau=1., kappa=0.):
    u, v = xsy2uv(x, s, y, tau, kappa)
    return u - v


@nb.jit(nopython=True)
def uv2xsytaukappa(u, v, n):
    tau = np.float(u[-1])
    kappa = np.float(v[-1])
    x = u[:n] / tau if tau > 0 else u[:n] / kappa
    y = u[n:-1] / tau if tau > 0 else u[n:-1] / kappa
    s = v[n:-1] / tau if tau > 0 else v[n:-1] / kappa
    return x, s, y, tau, kappa

# @nb.jit(nopython=True)


@nb.jit(nopython=True)
def z2uv(z, n, cones):
    u, cache = embedded_cone_Pi(z, *cones, n)
    return u, u - z, cache


@nb.jit(nopython=True)
def z2xsy(z, n, cones):
    # TODO implement infeasibility cert.
    u, cache = embedded_cone_Pi(z, *cones, n)
    v = u - z
    x, s, y, tau, kappa = uv2xsytaukappa(u, v, n)
    return x, s, y


CSC_mattypes = nb.types.Tuple((nb.int32[:], nb.int32[:], nb.float64[:]))

#@nb.jit(nopython=True)


@nb.jit(nb.float64[:](
    CSC_mattypes,
    # CSC_mattypes,
    nb.float64[:],
    nb.float64[:],
    nb.float64[:]), nopython=True)
def Q_matvec(A, b, c, u):
    col_pointers, row_indeces, mat_elements = A
    m, n = len(b), len(c)
    result = np.empty_like(u)
    result[:n] = csr_matvec(col_pointers, row_indeces, mat_elements, u[n:-1]) \
        + c * u[-1]
    # result[:n] = A_tr@u[n:-1] + c * u[-1]
    # result[n:-1] = -A@u[:n] + b * u[-1]
    result[n:-1] = - csc_matvec(m, col_pointers, row_indeces, mat_elements, u[:n])\
        + b * u[-1]
    result[-1] = -c.T@u[:n] - b.T@u[n:-1]
    return result


#@nb.jit(nopython=True)
@nb.jit(nb.float64[:](
    CSC_mattypes,  # CSC_mattypes,
    nb.float64[:],
    nb.float64[:],
    nb.float64[:]), nopython=True)
def Q_rmatvec(A, b, c, u):
    return -Q_matvec(A, b, c, u)


#@nb.jit(nopython=True)
@nb.jit(nb.float64[:](
    nb.float64[:],
    nb.float64[:],
    CSC_mattypes,  # CSC_mattypes,
    nb.float64[:],
    nb.float64[:],
    cache_types), nopython=True)
def residual_D(z, dz, A, b, c, cones_caches):
    m, n = len(b), len(c)
    zero, l, q, q_cache, s, s_cache_eivec, \
        s_cache_eival,  ep, \
        ep_cache, ed, ed_cache = cones_caches
    du = embedded_cone_D(z, dz, zero, l, q, q_cache, s, s_cache_eivec,
                         s_cache_eival,  ep,
                         ep_cache, ed, ed_cache, n)
    dv = du - dz
    return Q_matvec(A,  b, c, du) - dv


#@nb.jit(nopython=True)
@nb.jit(nb.float64[:](
    nb.float64[:],
    nb.float64[:],
    CSC_mattypes,  # CSC_mattypes,
    nb.float64[:],
    nb.float64[:],
    cache_types), nopython=True)
def residual_DT(z, dres, A,  b, c, cones_caches):
    m, n = len(b), len(c)
    zero, l, q, q_cache, s, s_cache_eivec, \
        s_cache_eival,  ep, \
        ep_cache, ed, ed_cache = cones_caches
    return embedded_cone_D(z, -Q_matvec(A, b, c, dres) - dres,
                           zero, l, q, q_cache, s, s_cache_eivec,
                           s_cache_eival,  ep,
                           ep_cache, ed, ed_cache, n) + dres


#@nb.jit(nopython=True)
@nb.jit(nb.types.Tuple((nb.float64[:], nb.float64[:], nb.float64[:]))(
    nb.float64[:],
    CSC_mattypes,  # CSC_mattypes,
    nb.float64[:],
    nb.float64[:],
    cache_types), nopython=True)
def residual_and_uv(z, A, b, c, cones_caches):
    m, n = len(b), len(c)
    zero, l, q, q_cache, s, s_cache_eivec, \
        s_cache_eival,  ep, \
        ep_cache, ed, ed_cache = cones_caches
    u = embedded_cone_Pi(z, zero, l, q, q_cache, s, s_cache_eivec,
                         s_cache_eival,  ep,
                         ep_cache, ed, ed_cache, n)
    v = u - z
    return Q_matvec(A, b, c, u) - v, u, v


def residual(z, A, b, c, cones_caches):
    A = sp.csc_matrix(A)
    A = (A.indptr, A.indices, A.data)
    res, u, v = residual_and_uv(z, A, b, c, cones_caches)
    return res


# @nb.jit()
# def print_header():  # z, norm_Q):
#     print()
#     print()
#     print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
#     print('              Conic Solution Refinement              ')
#     print()
#     print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
#     print("ite.    || N(z) ||_2      z[-1]     LSQR       time")
#     print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')


# # @nb.jit()
# # def subopt_stats(A, b, c, x, s, y):
# #     pri_res_norm = np.linalg.norm(
# #         A@x + s - b) / (1. + np.linalg.norm(b))
# #     dua_res_norm = np.linalg.norm(A.T@y + c) / \
# #         (1. + np.linalg.norm(c))
# #     rel_gap = np.abs(c@x + b@y) / \
# #         (1. + np.abs(c@x) + np.abs(b@y))
# #     compl_gap = s@y / (norm(s) * norm(y))
# #     return pri_res_norm, dua_res_norm, rel_gap, compl_gap


# # def print_stats(i, residual, residual_DT, num_lsqr_iters, start_time):
# @nb.jit()
# def print_stats(i, residual, z, num_lsqr_iters, start_time):
#     print('%d\t%.2e\t%.0e\t%d\t%.2f' %
#           (i, np.linalg.norm(residual / z[-1]), z[-1],
#            num_lsqr_iters,
#            time.time() - start_time))


# @nb.jit()
# def print_footer(message):
#     print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
#     print(message)
#     print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')


@nb.jit(nb.float64[:](
    nb.float64[:],
    nb.float64[:],
    CSC_mattypes,  # CSC_mattypes,
    nb.float64[:],
    nb.float64[:],
    cache_types,
    nb.float64[:]), nopython=True)
def lsqr_D(z, dz, A, b, c, cache, residual):
    return residual_D(z, dz, A, b, c, cache) / np.abs(z[-1]) \
        - np.sign(z[-1]) * (residual / z[-1]**2) * dz[-1]


@nb.jit(nb.float64[:](
    nb.float64[:],
    nb.float64[:],
    CSC_mattypes,  # CSC_mattypes,
    nb.float64[:],
    nb.float64[:],
    cache_types,
    nb.float64[:]), nopython=True)
def lsqr_DT(z, dres, A, b, c, cache, residual):
    m, n = len(b), len(c)
    e_minus1 = np.zeros(n + m + 1)
    e_minus1[-1] = 1.
    return residual_DT(z, dres, A, b, c, cache) / np.abs(z[-1]) \
        - np.sign(z[-1]) * (dres@residual / z[-1]**2) * e_minus1


# def lsqr_D(z, dz, A, b, c, cache, residual):
#     return residual_D(z, dz, A, b, c, cache)


# def lsqr_DT(z, dres, A, b, c, cache, residual):
#     return residual_DT(z, dres, A, b, c, cache)


# def normalized_resnorm(residual, z):
#     return np.linalg.norm(residual) / np.abs(z[-1])


# def refine(A, b, c, dim_dict, z,
#            iters=2,
#            lsqr_iters=30,
#            verbose=True):

#     m, n = A.shape

#     tmp = sp.csc_matrix(A)
#     A = (tmp.indptr, tmp.indices, tmp.data)

#     A_tr = sp.csc_matrix(tmp.T)
#     A_tr = (A_tr.indptr, A_tr.indices, A_tr.data)

#     start_time = time.time()

#     if verbose:
#         print_header()

#     cones_caches = make_prod_cone_cache(dim_dict)
#     residual, u, v = residual_and_uv(z, A, A_tr, b, c, cones_caches)
#     normres = np.linalg.norm(residual) / np.abs(z[-1])

#     refined_normres = float(normres)
#     refined = np.copy(z)

#     if verbose:
#         print_stats(0, residual, z, 0, start_time)

#     for i in range(iters):

#         if norm(lsqr_DT(z, residual, A, A_tr, b, c, cones_caches, residual)) == 0.:
#             if verbose:
#                 print_footer('Residual orthogonal to derivative.')
#             return refined

#         # D = LinearOperator((m + n + 1, m + n + 1),
#         #                    matvec=lambda dz: lsqr_D(
#         #     z, dz, A, A_tr, b, c, cones_caches, residual),
#         #     rmatvec=lambda dres: lsqr_DT(
#         #     z, dres, A, A_tr, b, c, cones_caches, residual)
#         # )
#         _ = lsqr(m + n + 1, m + n + 1,
#                  (z, A, A_tr, b, c, cones_caches, residual),
#                  # lambda dz: lsqr_D(z, dz, A, A_tr, b, c,
#                  #                   cones_caches, residual),
#                  # lambda dres: lsqr_DT(z, dres, A, A_tr, b,
#                  #                      c, cones_caches, residual),
#                  residual / np.abs(z[-1]), iter_lim=lsqr_iters)
#         step, num_lsqr_iters = _[0], _[2]
#         assert not True in np.isnan(step)

#         z = z - step
#         residual, u, v = residual_and_uv(
#             z, A, A_tr, b, c, cones_caches)
#         normres = np.linalg.norm(residual) / np.abs(z[-1])

#         if normres < refined_normres:
#             refined_normres = normres
#             refined = z / np.abs(z[-1])

#         if verbose:
#             print_stats(i + 1, np.copy(residual), np.copy(z),
#                         num_lsqr_iters, start_time)

#         if i == iters - 1:
#             if verbose:
#                 print_footer('Max num. refinement iters reached.')

#             return refined  # z / np.abs(z[-1])


# # def refine(A, b, c, dim_dict, z,
#            iters=2,
#            lsqr_iters=30,
#            max_backtrack=10,
#            verbose=True):

#     z = np.copy(z)  # / np.abs(z[-1])
#     m, n = A.shape

#     start_time = time.time()

#     if verbose:
#         print_header()  # z, sp.linalg.svds(Q(A, b, c), k=1)[1][0])

#     cones_caches = make_prod_cone_cache(dim_dict)
#     residual, u, v = residual_and_uv(z, A, b, c, cones_caches)
#     normres = np.linalg.norm(residual) / np.abs(z[-1])

#     if verbose:
#         print_stats(0, residual, z, 0, 0, start_time)

#     mu = np.zeros(len(residual))
#     theta = np.zeros(len(residual))
#     z_s = np.zeros((iters, len(residual)))
#     steps = np.zeros((iters, len(residual)))
#     residuals = np.zeros((iters, len(residual)))

#     for i in range(iters):

#         z_s[i] = z
#         residuals[i] = residual / np.abs(z[-1])

#         if norm(lsqr_DT(z, residual, A, b, c, cones_caches, residual)) == 0.:
#             if verbose:
#                 print_footer('Residual orthogonal to derivative.')
#             return z / np.abs(z[-1])

#         # residual, u, v = residual_and_uv(
#         #     z, A, b, c, cones_caches)

#         D = LinearOperator((m + n + 1, m + n + 1),
#                            matvec=lambda dz: lsqr_D(
#             z, dz, A, b, c, cones_caches, residual),
#             rmatvec=lambda dres: lsqr_DT(
#             z, dres, A, b, c, cones_caches, residual)
#         )
#         _ = lsqr(D, residual / np.abs(z[-1]),  # + mu / 2,
#                  damp=0.,  # 1E-8,
#                  atol=0.,
#                  btol=0.,
#                  show=False,
#                  iter_lim=lsqr_iters)
#         step, num_lsqr_iters = _[0], _[2]
#         assert not True in np.isnan(step)

#         steps[i] = - step

#         # def backtrack():
#         #     test_cone_cache = make_prod_cone_cache(dim_dict)
#         #     for j in range(max_backtrack):
#         #         test_z = z - step * 2**(-j)
#         #         test_z /= np.abs(test_z[-1])
#         #         test_res, u, v = residual_and_uv(
#         #             test_z, A, b, c, test_cone_cache)
#         #         test_normres = np.linalg.norm(test_res)  # / np.abs(test_z[-1])
#         #         if test_normres < normres:
#         #             return test_z, test_res, test_cone_cache, test_normres, j, True
#         #     return z, residual, cones_caches, normres, j,  False

#         #z, residual, cones_cache, normres, num_btrk, improved = backtrack()

#         # theta_t_minus_one = theta_t
#         # theta_t = z - step
#         # theta_t /= np.abs(theta_t[-1])
#         # z = theta_t + .2 * (theta_t - theta_t_minus_one)

#         # ANDERSON = 20

#         # def anderson():
#         #     import cvxpy as cvx
#         #     w = cvx.Variable(ANDERSON)
#         #     cvx.Problem(cvx.Minimize(cvx.norm(w @ residuals[i - ANDERSON + 1:i + 1])),
#         #                 [cvx.sum(w) == 1.]).solve()
#         #     print(w.value)
#         # return w.value @ (z_s[i - ANDERSON + 1:i + 1] + steps[i - ANDERSON +
#         # 1:i + 1])

#         # z = anderson() if i > ANDERSON else (z - step)
#         z = z - step
#         # theta_t_minus_one = theta_t
#         # theta_t = z - step
#         # z = theta_t + .1 * (theta_t - theta_t_minus_one)

#         # theta = .5 * theta + step
#         # z = z - theta

#         #z /= np.abs(z[-1])
#         residual, u, v = residual_and_uv(z, A, b, c, cones_caches)
#         normres = np.linalg.norm(residual) / np.abs(z[-1])
#         num_btrk = 0

#         #mu += (1 / np.e) * residual

#         # if not improved:
#         #     if verbose:
#         #         print_footer('Hit maximum number of backtracks.')
#         #     return z / np.abs(z[-1])

#         if verbose:
#             print_stats(i + 1, np.copy(residual), np.copy(z),
#                         num_lsqr_iters, num_btrk, start_time)

#         if i == iters - 1:
#             if verbose:
#                 print_footer('Max num. refinement iters reached.')

#             return z / np.abs(z[-1])  # , z_s, residuals
