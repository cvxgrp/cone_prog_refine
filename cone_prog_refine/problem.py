"""
Enzo Busseti, Walaa Moursi, Stephen Boyd, 2018

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
import numpy as np
from numpy.linalg import norm
import scipy.sparse as sp
import numba as nb

from scipy.sparse.linalg import lsqr, LinearOperator

from .cones import prod_cone
from .utils import *


# @nb.jit(nb.types.UniTuple(nb.double[:], 2)(
#     nb.double[:], nb.double[:], nb.double[:],
#     nb.optional(nb.double), nb.optional(nb.double)),
#     nopython=True)
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


@nb.jit(nopython=True)
def xsy2z(x, s, y, tau=1., kappa=0.):
    u, v = xsy2uv(x, s, y, tau, kappa)
    return u - v

# @nb.jit(nopython=True)


def uv2xsytaukappa(u, v, n):
    tau = np.float(u[-1])
    kappa = np.float(v[-1])
    x = np.array(u[:n]) / tau if tau > 0 else kappa
    y = np.array(u[n:-1]) / tau if tau > 0 else kappa
    s = np.array(v[n:-1]) / tau if tau > 0 else kappa
    return x, s, y, tau, kappa

# @nb.jit(nopython=True)


def z2uv(z, n, cones):
    u, cache = embedded_cone_Pi(z, *cones, n)
    return u, u - z, cache


def z2xsy(z, n, cones):
    # TODO implement infeasibility cert.
    u, cache = embedded_cone_Pi(z, *cones, n)
    v = u - z
    x, s, y, tau, kappa = uv2xsytaukappa(u, v, n)
    return x, s, y


def Q_matvec(A, b, c, u):
    m, n = A.shape
    if len(u.shape) > 0:
        u = u.flatten()
    result = np.empty_like(u)
    result[:n] = A.T@u[n:-1] + c * u[-1]
    result[n:-1] = -A@u[:n] + b * u[-1]
    result[-1] = -c.T@u[:n] - b.T@u[n:-1]
    return result


def Q_rmatvec(A, b, c, u):
    return -Q_matvec(A, b, c, u)


def Q(A, b, c):
    m, n = A.shape
    return LinearOperator(shape=(n + m + 1, n + m + 1),
                          matvec=lambda u: Q_matvec(A, b, c, u),
                          rmatvec=lambda v: - Q_matvec(A, b, c, v))


def norm_Q(A, b, c):
    return sp.linalg.svds(Q(A, b, c), k=1)[1][0]


def residual_D(z, dz, A, b, c, cones_caches):
    m, n = A.shape
    du = embedded_cone_D(z, dz, *cones_caches, n)
    dv = du - dz
    return Q_matvec(A, b, c, du) - dv


def residual_DT(z, dres, A, b, c, cones_caches):
    m, n = A.shape
    return embedded_cone_D(z, -Q_matvec(A, b, c, dres) - dres,
                           *cones_caches, n) + dres


def residual_and_uv(z, A, b, c, cones_caches):
    m, n = A.shape
    u = embedded_cone_Pi(z, *cones_caches, n)
    v = u - z
    return Q_matvec(A, b, c, u) - v, u, v


def residual(z, A, b, c, cones_caches):
    res, u, v = residual_and_uv(z, A, b, c, cones_caches)
    return res


# def scs_solve(A, b, c, dim_dict, **kwargs):
#     """Wraps scs.solve for convenience."""
#     scs_cones = {'l': dim_dict['l'] if 'l' in dim_dict else 0,
#                  'q': dim_dict['q'] if 'q' in dim_dict else [],
#                  's': dim_dict['s'] if 's' in dim_dict else [],
#                  'ep': dim_dict['ep'] if 'ep' in dim_dict else 0,
#                  'ed': dim_dict['ed'] if 'ed' in dim_dict else 0,
#                  'f': dim_dict['z'] if 'z' in dim_dict else 0}
#     #print('scs_cones', scs_cones)
#     sol = scs.solve({'A': A, 'b': b,
#                      'c': c},
#                     cone=scs_cones,
#                     use_indirect=True,
#                     # cg_rate=2.,
#                     normalize=True,
#                     scale=1.,
#                     **kwargs)
#     info = sol['info']

#     if info['statusVal'] > 0:
#         z = xsy2z(sol['x'], sol['s'], sol['y'], tau=1., kappa=0.)

#     if info['statusVal'] < 0:
#         sol['x'] = np.zeros_like(sol['x']) \
#             if np.any(np.isnan(sol['x'])) else sol['x']

#         sol['s'] = np.zeros_like(sol['s']) \
#             if np.any(np.isnan(sol['s'])) else sol['s']

#         sol['y'] = np.zeros_like(sol['y']) \
#             if np.any(np.isnan(sol['y'])) else sol['y']

#         z = xsy2z(sol['x'], sol['s'], sol['y'], tau=0., kappa=1.)

#     return z, info


# def ecos_solve(A, b, c, dim_dict, **kwargs):
#     """Wraps ecos.solve for convenience."""
#     ecos_cones = {'l': dim_dict['l'] if 'l' in dim_dict else 0,
#                   'q': dim_dict['q'] if 'q' in dim_dict else [],
#                   'e': dim_dict['ep'] if 'ep' in dim_dict else 0}
#     # print(ecos_cones)
#     # TODO check and block other cones
#     zero = 0 if 'z' not in dim_dict else dim_dict['z']
#     ecos_A, ecos_G = A[:zero, :], A[zero:, :]
#     ecos_b, ecos_h = b[:zero], b[zero:]
#     sol = ecos.solve(c=c, G=ecos_G, h=ecos_h, dims=ecos_cones,
#                      A=ecos_A, b=ecos_b, **kwargs)

#     z = xsy2z(sol['x'],
#               np.concatenate([np.zeros(zero), sol['s']]),
#               np.concatenate([sol['y'], sol['z']]),
#               tau=1., kappa=0.)

#     return z, sol['info']


def print_header():  # z, norm_Q):
    print()
    print()
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('               Conic Solution Refinement              ')
    print()
    print("it.    ||N(z)||_2       z[-1]   LSQR  btrks     time")
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')


def subopt_stats(A, b, c, x, s, y):
    pri_res_norm = np.linalg.norm(
        A@x + s - b) / (1. + np.linalg.norm(b))
    dua_res_norm = np.linalg.norm(A.T@y + c) / \
        (1. + np.linalg.norm(c))
    rel_gap = np.abs(c@x + b@y) / \
        (1. + np.abs(c@x) + np.abs(b@y))
    compl_gap = s@y / (norm(s) * norm(y))
    return pri_res_norm, dua_res_norm, rel_gap, compl_gap


# def print_stats(i, residual, residual_DT, num_lsqr_iters, start_time):
def print_stats(i, residual, z, num_lsqr_iters, backtracks, start_time):
    print('%d\t%.2e\t%.0e\t  %d\t%d\t%.2f' %
          (i, np.linalg.norm(residual / z[-1]), z[-1],
           num_lsqr_iters, backtracks,
           time.time() - start_time))


def print_footer(message):
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print(message)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')


# def lsqr_D(z, dz, A, b, c, cache, residual):
#     return residual_D(z, dz, A, b, c, cache) / z[-1] - (residual /
#                                                         z[-1]**2) * dz[-1]
#
#
# def lsqr_DT(z, dres, A, b, c, cache, residual):
#     m, n = A.shape
#     e_minus1 = np.zeros(n + m + 1)
#     e_minus1[-1] = 1.
#     return residual_DT(z, dres, A, b, c, cache) / z[-1] - (dres@residual /
#                                                            z[-1]**2) * e_minus1
#

def lsqr_D(z, dz, A, b, c, cache, residual):
    return residual_D(z, dz, A, b, c, cache)


def lsqr_DT(z, dres, A, b, c, cache, residual):
    return residual_DT(z, dres, A, b, c, cache)


def normalized_resnorm(residual, z):
    return np.linalg.norm(residual / z[-1])


def refine(A, b, c, dim_dict, z,
           iters=2,
           lsqr_iters=30,
           max_backtrack=10,
           verbose=True):

    z = np.copy(z) / np.abs(z[-1])
    m, n = A.shape

    start_time = time.time()

    if verbose:
        print_header()  # z, sp.linalg.svds(Q(A, b, c), k=1)[1][0])

    cones_caches = make_prod_cone_cache(dim_dict)
    residual, u, v = residual_and_uv(z, A, b, c, cones_caches)
    normres = np.linalg.norm(residual)

    if verbose:
        print_stats(0, residual, z, 0, 0, start_time)

    for i in range(iters):

        if norm(residual_DT(z, residual, A, b, c, cones_caches)) == 0.:
            if verbose:
                print_footer('Residual orthogonal to derivative.')
            return z / np.abs(z[-1])

        # residual, u, v = residual_and_uv(
        #     z, A, b, c, cones_caches)

        D = LinearOperator((m + n + 1, m + n + 1),
                           matvec=lambda dz: lsqr_D(
            z, dz, A, b, c, cones_caches, residual),
            rmatvec=lambda dres: lsqr_DT(
            z, dres, A, b, c, cones_caches, residual)
        )
        _ = lsqr(D, residual,
                 damp=1E-8,
                 atol=0.,
                 btol=0.,
                 show=False,
                 iter_lim=lsqr_iters)
        step, num_lsqr_iters = _[0], _[2]
        assert not True in np.isnan(step)

        def backtrack():
            test_cone_cache = make_prod_cone_cache(dim_dict)
            for j in range(max_backtrack):
                test_z = z - step * 2**(-j)
                test_z /= np.abs(test_z[-1])
                test_res, u, v = residual_and_uv(test_z, A, b, c, test_cone_cache)
                test_normres = np.linalg.norm(test_res)
                if test_normres < normres:
                    return test_z, test_res, test_cone_cache, test_normres, j, True
            return z, residual, cones_caches, normres, j,  False

        z, residual, cones_cache, normres, num_btrk, improved = backtrack()

        if not improved:
            if verbose:
                print_footer('Hit maximum number of backtracks.')
            return z / np.abs(z[-1])

        if verbose:
            print_stats(i + 1, np.copy(residual), np.copy(z),
                        num_lsqr_iters, num_btrk, start_time)

        if i == iters - 1:
            if verbose:
                print_footer('Max num. refinement iters reached.')

            return z / np.abs(z[-1])
