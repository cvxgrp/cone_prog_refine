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
from scipy.sparse.linalg import LinearOperator

from .lsqr import lsqr

from .cones import prod_cone, free_cone, zero_cone,\
    non_neg_cone, sec_ord_cone, semi_def_cone

from .utils import *

from .jit import jit, njit


@jit
def xsy2uv(x, s, y, tau=1., kappa=0.):
    return np.concatenate([x, y, [tau]]), \
        np.concatenate([np.zeros(len(x)), s, [kappa]])


@jit
def xsy2z(x, s, y, tau=1., kappa=0.):
    u, v = xsy2uv(x, s, y, tau, kappa)
    return u - v


@jit
def uv2xsytaukappa(u, v, n):
    tau = np.float(u[-1])
    kappa = np.float(v[-1])
    x = np.array(u[:n]) / tau if tau > 0 else kappa
    y = np.array(u[n:-1]) / tau if tau > 0 else kappa
    s = np.array(v[n:-1]) / tau if tau > 0 else kappa
    return x, s, y, tau, kappa


@jit
def z2uv(z, n, cones):
    u, cache = embedded_cone_Pi(z, cones, n)
    return u, u - z, cache


@jit
def z2xsy(z, n, cones):
    # TODO implement infeasibility cert.
    u, cache = embedded_cone_Pi(z, cones, n)
    v = u - z
    x, s, y, tau, kappa = uv2xsytaukappa(u, v, n)
    return x, s, y


@jit
def Q_matvec(A, b, c, u):
    m, n = A.shape
    if len(u.shape) > 0:
        u = u.flatten()
    result = np.empty_like(u)
    result[:n] = A.T@u[n:-1] + c * u[-1]
    result[n:-1] = -A@u[:n] + b * u[-1]
    result[-1] = -c.T@u[:n] - b.T@u[n:-1]
    return result


@jit
def Q_rmatvec(A, b, c, u):
    return -Q_matvec(A, b, c, u)


# def Q(A, b, c):
#     m, n = A.shape
#     return LinearOperator(shape=(n + m + 1, n + m + 1),
#                           matvec=lambda u: Q_matvec(A, b, c, u),
#                           rmatvec=lambda v: - Q_matvec(A, b, c, v))


@jit
def norm_Q(A, b, c):
    return sp.linalg.svds(Q(A, b, c), k=1)[1][0]


@jit
def residual_D(z, dz, A, b, c, cones_caches):
    m, n = A.shape
    du = embedded_cone_D(z, dz, cones_caches, n)
    dv = du - dz
    return Q_matvec(A, b, c, du) - dv


@jit
def residual_DT(z, dres, A, b, c, cones_caches):
    m, n = A.shape
    return embedded_cone_D(z, -Q_matvec(A, b, c, dres) - dres,
                           cones_caches, n) + dres


@jit
def residual_and_uv(z, A, b, c, cones_caches):
    m, n = A.shape
    u = embedded_cone_Pi(z, cones_caches, n)
    v = u - z
    return Q_matvec(A, b, c, u) - v, u, v


@jit
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


@jit
def print_header(z, norm_Q):
    print()
    print()
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print('               Conic Solution Refinement              ')
    print('            E. Busseti, W. Moursi, S. Boyd            ')
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print(' len(z) = %d,  ||z|| = %.2e,  ||Q||_2 = %.2e ' %
          (len(z), np.linalg.norm(z), norm_Q))
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print("it.   ||R(z)/z[-1]||_2    z[-1]   LSQR   btrks    time")
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')


@jit
def subopt_stats(A, b, c, x, s, y):
    pri_res_norm = np.linalg.norm(
        A@x + s - b) / (1. + np.linalg.norm(b))
    dua_res_norm = np.linalg.norm(A.T@y + c) / \
        (1. + np.linalg.norm(c))
    rel_gap = np.abs(c@x + b@y) / \
        (1. + np.abs(c@x) + np.abs(b@y))
    compl_gap = s@y / (norm(s) * norm(y))
    return pri_res_norm, dua_res_norm, rel_gap, compl_gap


@jit
# def print_stats(i, residual, residual_DT, num_lsqr_iters, start_time):
def print_stats(i, residual, z, num_lsqr_iters, backtracks, start_time):
    print('%d\t%.2e\t%.0e\t  %d\t%d\t%.2f' %
          (i, np.linalg.norm(residual / z[-1]), z[-1],
           num_lsqr_iters, backtracks,
           time.time() - start_time))


@jit
def print_footer(message):
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
    print(message)
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')


@jit
def lsqr_D(z, dz, A, b, c, cache, residual):
    return residual_D(z, dz, A, b, c, cache) / z[-1] - (residual /
                                                        z[-1]**2) * dz[-1]


@jit
def lsqr_DT(z, dres, A, b, c, cache, residual):
    m, n = A.shape
    e_minus1 = np.zeros(n + m + 1)
    e_minus1[-1] = 1.
    return residual_DT(z, dres, A, b, c, cache) / z[-1] - (dres@residual /
                                                           z[-1]**2) * e_minus1


# def lsqr_D(z, dz, A, b, c, cache, residual):
#     return residual_D(z, dz, A, b, c, cache)


# def lsqr_DT(z, dres, A, b, c, cache, residual):
#     return residual_DT(z, dres, A, b, c, cache)


@jit
def normalized_resnorm(residual, z):
    return np.linalg.norm(residual / z[-1])


@jit
def backtrack(z, res, normres, step, A, b, c, cones, max_iters):

    for j in range(max_iters):

        test_z = z - step * 2**(-j)
        test_z /= np.abs(test_z[-1])

        test_res, u, v, test_cache = residual_and_uv(
            test_z, A, b, c, cones)
        test_normres = np.linalg.norm(test_res)
        if test_normres < normres:
            return test_z, test_res, test_normres, test_cache, j, False

    return z, res, normres, None, j, True


@jit
def refine(A, b, c, cones, z,
           iters=2,
           lsqr_iters=30,
           max_backtrack=10,
           verbose=True):

    z = np.copy(z) / np.abs(z[-1])
    m, n = A.shape

    if verbose:
        print_header(z, sp.linalg.svds(Q(A, b, c), k=1)[1][0])

    res, u, v, cache = residual_and_uv(z, A, b, c, cones)
    normres = np.linalg.norm(res)

    start_time = time.time()

    if verbose:
        print_stats(0, res, z, 0, 0, start_time)

    for i in range(iters):

        if norm(residual_DT(z, res, A, b, c, cache)) == 0.:
            if verbose:
                print_footer('Residual orthogonal to derivative.')
            return z / np.abs(z[-1])

        _ = lsqr(A, b, c, cones, z, res,
                 damp=1E-8,
                 atol=0.,
                 btol=0.,
                 show=False,
                 iter_lim=lsqr_iters)
        step, num_lsqr_iters = _[0], _[2]

        assert not True in np.isnan(step)

        z, res, normres, cache, backtracks, failed = backtrack(
            z, res, normres, step,
            A, b, c, cones,
            max_iters=max_backtrack)

        # print('normres returned by BT %.2e' % normres)
        # res, u, v, cache = residual_and_uv(z, A, b, c, cones)
        # normres = normalized_resnorm(res,  z)
        # print('recomputed normres %.2e' % normres)

        if failed:
            if verbose:
                print_footer('Hit maximum number of backtracks.')
            return z / np.abs(z[-1])

        # backtracks = 0
        # for j in range(max_backtrack):
        #     test_z = z - step * 2**(-i)
        #     test_res, u, v, test_cache = residual_and_uv(
        #         test_z, A, b, c, cones)
        #     test_normres = np.linalg.norm(test_res) / np.abs(test_z[-1])
        #     if test_normres < normres:
        #         break

        # new_z = z - step
        # # new_z /= np.abs(new_z[-1])
        # new_residual, u, v, new_cache = residual_and_uv(new_z, A, b, c, cones)

        # backtracks = 0
        # # backtracking
        # while np.linalg.norm(new_residual / new_z[-1]) > np.linalg.norm(residual / z[-1]):
        #     # print('backtracking')
        #     step /= 2.
        #     backtracks += 1
        #     new_z = z - step
        #     new_residual, u, v, new_cache = residual_and_uv(
        #         new_z, A, b, c, cones)

        # # try one more divide by 2
        # test_z = z - step / 2.
        # test_residual, u, v, test_cache = residual_and_uv(
        #     test_z, A, b, c, cones)
        # if np.linalg.norm(test_residual / test_z[-1]) < np.linalg.norm(new_residual / new_z[-1]):
        #     #print('swapping with shorter step')
        #     new_z = test_z
        #     new_residual = test_residual
        #     new_cache = test_cache
        #     backtracks += 1

        # print('here i make stuff to go to screen')
        # myz = new_z / np.abs(new_z[-1])
        # myresidual, u, v, _ = residual_and_uv(
        #     myz, A, b, c, cones)
        # print('norm of vec returned %.2e' %
        #       np.linalg.norm(myresidual))

        # if verbose:
        #     print_stats(i + 1, new_residual, new_z,
        #                 num_lsqr_iters, backtracks, start_time)
        if verbose:
            print_stats(i + 1, np.copy(res), np.copy(z),
                        num_lsqr_iters, backtracks, start_time)

        # rel_res_change = (np.linalg.norm(residual / z[-1]) - np.linalg.norm(
        #     new_residual / new_z[-1])) / np.linalg.norm(residual / z[-1])

        # old_z = np.copy(z)

        # cache = new_cache
        # z = new_z
        # residual = new_residual

        # if rel_res_change < 1E-8:
        #     if verbose:
        #         print_footer('Residual change too small.')
        #     return z / np.abs(z[-1])

        if i == iters - 1:
            if verbose:
                print_footer('Max num. refinement iters reached.')

            # print('normres after print %.2e' % normres)
            # res, u, v, cache = residual_and_uv(z, A, b, c, cones)
            # normres = normalized_resnorm(res,  z)
            # print('recomputed normres %.2e' % normres)

            # myz = z / np.abs(z[-1])
            # new_residual, u, v, new_cache = residual_and_uv(
            #     myz, A, b, c, cones)
            # print('norm of vec returned %.2e' %
            #       np.linalg.norm(new_residual))
            return z / np.abs(z[-1])
