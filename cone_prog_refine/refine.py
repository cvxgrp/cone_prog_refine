"""
Enzo Busseti, Walaa Moursi, Stephen Boyd, 2019

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
import time
import numba as nb
import numpy as np
import scipy.sparse as sp
from .cones import *
from .lsqr import lsqr

from .problem import residual_and_uv


#@nb.jit((nb.types.float64[:], ))
def print_header(entries_of_A, b, c, dim_dict):  # z, norm_Q):
    n = len(b) + len(c) + 1
    print()
    print()
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('           Cone Program Solution Refinement          ')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print()
    #print('Primal-dual embedded problem with matrix Q and cone 𝒦.')
    print('Q ∈ 𝗥^(%d × %d),  nnz(Q) = %d,  ‖Q‖_F = %.3e,' % (n, n,
                                                             len(entries_of_A),
                                                             np.sqrt(sum(entries_of_A**2) +
                                                                     sum(b**2) + sum(c)**2)))
    print()
    #cone_string = 'and cone 𝒦 ∈ 𝗥^%d,\n' % n
    cone_string = '𝒦 = 𝗥^(%d)' % (len(c) + (dim_dict['z']
                                            if 'z' in dim_dict else 0))
    cone_string += ' × 𝗥_+^(%d)' % (1 +
                                    (dim_dict['l'] if 'l' in dim_dict else 0))
    cone_string += (' × 𝓢^(%d)' %
                    sum(dim_dict['q'])) if 'q' in dim_dict else ''
    cone_string += (' × 𝗦𝗗^(%d)' %
                    sum((np.array(dim_dict['s'])**2 +
                         np.array(dim_dict['s'])) // 2)) if 's' in dim_dict else ''
    cone_string += (' × 𝒦_exp^(%d)' %
                    dim_dict['ed']) if 'ed' in dim_dict else ''
    cone_string += (' × 𝒦_exp^*(%d)' %
                    dim_dict['ep']) if 'ep' in dim_dict else ''
    print(cone_string)
    print()
    if 'q' in dim_dict:
        print('𝓢 is the product of %d second-order cones' %
              len(dim_dict['q']))
    if 's' in dim_dict:
        print('𝗦𝗗 is the product of %d semi-definite cones' %
              len(dim_dict['s']))
    if 'ep' in dim_dict or 'ed' in dim_dict:
        print('each 𝒦_exp ∈ 𝗥^3 is an exponential cone')
    print()
    print('           (Enzo Busseti 𝘦𝘵 𝘢𝘭., 2017-2019)            ')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print("ite.   ‖𝒩(z^+)‖   ‖𝒩(z) + 𝗗𝛿‖   z_(n-1)  LSQR   time")
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')


# @nb.jit()
# def subopt_stats(A, b, c, x, s, y):
#     pri_res_norm = np.linalg.norm(
#         A@x + s - b) / (1. + np.linalg.norm(b))
#     dua_res_norm = np.linalg.norm(A.T@y + c) / \
#         (1. + np.linalg.norm(c))
#     rel_gap = np.abs(c@x + b@y) / \
#         (1. + np.abs(c@x) + np.abs(b@y))
#     compl_gap = s@y / (norm(s) * norm(y))
#     return pri_res_norm, dua_res_norm, rel_gap, compl_gap


# def print_stats(i, residual, residual_DT, num_lsqr_iters, start_time):
@nb.jit()
def print_stats(i, residual, r1norm, z, num_lsqr_iters, start_time):
    print('%d\t%.2e\t%.2e\t%.0e\t%d\t%.2f' %
          (i, np.linalg.norm(residual / z[-1]), r1norm, z[-1],
           num_lsqr_iters,
           time.time() - start_time))


@nb.jit()
def print_stats_ADMM(i, residual, u, z, num_lsqr_iters, start_time):
    print('%d\t%.2e\t%.2e\t%.0e\t\t%d\t%.2f' %
          (i, np.linalg.norm(residual / z[-1]),
           np.linalg.norm(u),
           z[-1],
           num_lsqr_iters,
           time.time() - start_time))


@nb.jit()
def print_footer(message):
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print(message)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')


#@nb.jit(nopython=True)
def refine(A, b, c, dim_dict, z,
           iters=2,
           lsqr_iters=30,
           verbose=True):

    m, n = A.shape

    A = sp.csc_matrix(A)
    A = (A.indptr, A.indices, A.data)

    # A_tr = sp.csc_matrix(tmp.T)
    # A_tr = (A_tr.indptr, A_tr.indices, A_tr.data)

    start_time = time.time()

    if verbose:
        print_header(A[2], b, c, dim_dict)

    cones_caches = make_prod_cone_cache(dim_dict)
    residual, u, v = residual_and_uv(z, A, b, c, cones_caches)
    normres = np.linalg.norm(residual) / np.abs(z[-1])

    refined_normres = float(normres)
    refined = np.copy(z)

    if verbose:
        print_stats(0, residual, np.nan, z, 0, start_time)

    for i in range(iters):

        # if norm(lsqr_DT(z, residual, A, A_tr, b, c, cones_caches, residual)) == 0.:
        #     if verbose:
        #         print_footer('Residual orthogonal to derivative.')
        #     return refined

        # D = LinearOperator((m + n + 1, m + n + 1),
        #                    matvec=lambda dz: lsqr_D(
        #     z, dz, A, A_tr, b, c, cones_caches, residual),
        #     rmatvec=lambda dres: lsqr_DT(
        #     z, dres, A, A_tr, b, c, cones_caches, residual)
        # )

        # lambda dz: lsqr_D(z, dz, A, A_tr, b, c,
        #                   cones_caches, residual),
        # lambda dres: lsqr_DT(z, dres, A, A_tr, b,
        #                      c, cones_caches, residual),

        step, num_lsqr_iters, r1norm, acond = lsqr(m=m + n + 1,
                                                   n=m + n + 1,
                                                   operator_vars=(
                                                       z, A, b, c, cones_caches, residual),
                                                   b=residual / np.abs(z[-1]),
                                                   iter_lim=lsqr_iters)
        # step, num_lsqr_iters = _[0], _[2]
        assert not True in np.isnan(step)

        z = z - step
        residual, u, v = residual_and_uv(
            z, A, b, c, cones_caches)
        normres = np.linalg.norm(residual) / np.abs(z[-1])

        if normres < refined_normres:
            refined_normres = normres
            refined = z / np.abs(z[-1])

        if verbose:
            print_stats(i + 1, np.copy(residual), r1norm, np.copy(z),
                        num_lsqr_iters, start_time)

        if i == iters - 1:
            if verbose:
                print_footer('Max num. refinement iters reached.')

            return refined  # z / np.abs(z[-1])


def ADMM_refine(A, b, c, dim_dict, z,
                iters=2,
                lsqr_iters=30,
                verbose=True,
                U_UPDATE=20):

    m, n = A.shape

    A = sp.csc_matrix(A)
    A = (A.indptr, A.indices, A.data)

    start_time = time.time()

    if verbose:
        print_header(A[2], b, c, dim_dict)

    cones_caches = make_prod_cone_cache(dim_dict)
    residual, u, v = residual_and_uv(z, A, b, c, cones_caches)
    normres = np.linalg.norm(residual) / np.abs(z[-1])

    refined_normres = float(normres)
    refined = np.copy(z)

    if verbose:
        print_stats(0, residual, z, 0, start_time)

    my_u = np.zeros(len(z))

    for i in range(iters):

        step, num_lsqr_iters, r1norm, acond = lsqr(m=m + n + 1,
                                                   n=m + n + 1,
                                                   operator_vars=(
                                                       z, A, b, c, cones_caches, residual),
                                                   b=(residual /
                                                      np.abs(z[-1])) + my_u,
                                                   iter_lim=lsqr_iters)
        assert not True in np.isnan(step)

        z = z - step
        residual, u, v = residual_and_uv(
            z, A, b, c, cones_caches)
        normres = np.linalg.norm(residual) / np.abs(z[-1])
        normalized_residual = residual / np.abs(z[-1])

        if i % U_UPDATE == (U_UPDATE - 1):
            my_u += normalized_residual

        # print('norm mu:', np.linalg.norm(mu))
        # print('normres:', normres)
        # print('refined_normres:', refined_normres)

        if normres < refined_normres:
            refined_normres = normres
            refined = z / np.abs(z[-1])

        if verbose:
            print_stats_ADMM(i + 1, np.copy(residual), np.copy(my_u), np.copy(z),
                             num_lsqr_iters, start_time)

        if i == iters - 1:
            if verbose:
                print_footer('Max num. refinement iters reached.')

            return refined  # z / np.abs(z[-1])
