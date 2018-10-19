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

from numba import jit, njit


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
    x = np.array(u[:n]) / tau
    y = np.array(u[n:-1]) / tau
    s = np.array(v[n:-1]) / tau
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


def Q(A, b, c):
    m, n = A.shape
    return LinearOperator(shape=(n + m + 1, n + m + 1),
                          matvec=lambda u: Q_matvec(A, b, c, u),
                          rmatvec=lambda v: - Q_matvec(A, b, c, v))


@jit
def norm_Q(A, b, c):
    return sp.linalg.svds(Q(A, b, c), k=1)[1][0]


@jit
def residual_D(z, dz, A, b, c, cones_caches):
    du = embedded_cone.D(z, dz, cones_caches)
    dv = du - dz
    return Q_matvec(A, b, c, du) - dv


@jit
def residual_DT(z, dres, A, b, c, cones_caches):
    return embedded_cone.DT(z,
                            -Q_matvec(A, b, c, dres) - dres,
                            cones_caches) + dres


@jit
def residual_and_uv(z, A, b, c, cones):
    m, n = A.shape
    u, cache = embedded_cone.Pi(z, cones, n)
    v = u - z
    return Q_matvec(A, b, c, u) - v, u, v, cache


#@jit
def residual(z, A, b, c, cones):
    res, u, v, cache = residual_and_uv(z, A, b, c, cones)
    return res, cache


def scs_solve(A, b, c, dim_dict, **kwargs):
    """Wraps scs.solve for convenience."""
    sol = scs.solve({'A': A, 'b': b,
                     'c': c},
                    cone=dim_dict,
                    use_indirect=True,
                    # cg_rate=2.,
                    normalize=True,
                    scale=1.,
                    **kwargs)
    info = sol['info']

    if info['statusVal'] > 0:
        z = xsy2z(sol['x'], sol['s'], sol['y'], tau=1., kappa=0.)

    if info['statusVal'] < 0:
        sol['x'] = np.zeros_like(sol['x']) \
            if np.any(np.isnan(sol['x'])) else sol['x']

        sol['s'] = np.zeros_like(sol['s']) \
            if np.any(np.isnan(sol['s'])) else sol['s']

        sol['y'] = np.zeros_like(sol['y']) \
            if np.any(np.isnan(sol['y'])) else sol['y']

        z = xsy2z(sol['x'], sol['s'], sol['y'], tau=0., kappa=1.)

    return z, info


def ecos_solve(A, b, c, dim_dict, **kwargs):
    """Wraps ecos.solve for convenience."""
    ecos_cones = {'l': dim_dict['l'] if 'l' in dim_dict else 0,
                  'q': dim_dict['q'] if 'q' in dim_dict else [],
                  'e': dim_dict['ep'] if 'ep' in dim_dict else 0}
    # TODO check and block other cones
    zero = 0 if 'z' not in dim_dict else dim_dict['z']
    ecos_A, ecos_G = A[:zero, :], A[zero:, :]
    ecos_b, ecos_h = b[:zero], b[zero:]
    sol = ecos.solve(c=c, G=ecos_G, h=ecos_h, dims=ecos_cones,
                     A=ecos_A, b=ecos_b, **kwargs)

    z = xsy2z(sol['x'],
              np.concatenate([np.zeros(zero), sol['s']]),
              np.concatenate([sol['y'], sol['z']]),
              tau=1., kappa=0.)

    return z, sol['info']


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
def refine(A, b, c, cones, z,
           iters=10,
           lsqr_iters=20,
           verbose=True,
           max_runtime=1.):

    #btol = .5

    m, n = A.shape

    lambda_var = 1.
    norm_Q = sp.linalg.svds(Q(A, b, c), k=1)[1][0]

    if verbose:
        print_header(z, norm_Q)

    # z /= (np.linalg.norm(z) / len(z))

    residual, u, v, cache = residual_and_uv(z, A, b, c, cones)
    start_time = time.time()

    if verbose:
        print_stats(0, residual, z,
                    # lambda dres: residual_DT(z, dres, A, b, c, cache),
                    0, 0, start_time)

    # mem_anderson = 2

    # samples_an = np.zeros((len(z), mem_anderson))  # [np.copy(z)]
    # residuals_an = np.zeros((len(z), mem_anderson))

    #start_z[:, i % mem_anderson] = z
    # start_residual = []  # [np.copy(residual)]

    #end_z = np.zeros((len(z), mem_anderson))
    #end_residual = []

    # mem_anderson_step = 5
    # steps_anderson = np.zeros((len(z), mem_anderson_step))
    # arrivals_anderson = np.zeros((len(z), mem_anderson_step))

    for i in range(iters):

        # if i >= mem_anderson_step:
        #     import cvxpy as cvx
        #     w = cvx.Variable(mem_anderson_step)
        #     obj = cvx.Minimize(cvx.norm(steps_anderson @ w))
        #     const = [cvx.sum(w) == 1.]
        #     cvx.Problem(obj, const).solve(verbose=False)

        #     anderson_z = (arrivals_anderson - steps_anderson / 2.) @ w.value
        #     anderson_residual, u, v, anderson_cache = residual_and_uv(
        #         anderson_z, A, b, c, cones)

        #     an_norm = np.linalg.norm(anderson_residual / anderson_z[-1])
        #     cur_norm = np.linalg.norm(residual / z[-1])
        #     print('an norm / cur_norm', (an_norm / cur_norm))

        #     # while np.linalg.norm(anderson_residual / anderson_z[-1]) > \
        #     #         np.linalg.norm(residual / z[-1]):
        #     #     print('backtracking')
        #     #     anderson_z = z + (anderson_z - z) / 2
        #     #     anderson_residual, u, v, anderson_cache = residual_and_uv(
        #     #         anderson_z, A, b, c, cones)

        #     if np.linalg.norm(anderson_residual / anderson_z[-1]) < cur_norm:
        #         print('swapping with anderson')
        #         z = anderson_z
        #         residual = anderson_residual
        #         cache = anderson_cache

        # samples_an[:, i % mem_anderson] = z
        # residuals_an[:, i % mem_anderson] = residual
        # start_residual.append(np.copy(residual))

        # if i > mem_anderson - 1:
        #     import cvxpy as cvx
        #     w = cvx.Variable(mem_anderson)
        #     obj = cvx.Minimize(cvx.sum_squares(residuals_an @ w))
        #     const = [cvx.sum(w) == 1.]
        #     cvx.Problem(obj, const).solve(verbose=False)

        #     anderson_z = samples_an @ w.value
        #     anderson_residual, u, v, anderson_cache = residual_and_uv(
        #         anderson_z, A, b, c, cones)

        #     an_norm = np.linalg.norm(anderson_residual / anderson_z[-1])
        #     cur_norm = np.linalg.norm(residual / z[-1])
        #     print('an norm / cur_norm', (an_norm/cur_norm))

        #     if an_norm < np.linalg.norm(residual / z[-1]):
        #         print('swapping with anderson')
        #         z = anderson_z
        #         residual = anderson_residual
        #         cache = anderson_cache

        if norm(residual_DT(z, residual, A, b, c, cache)) == 0.:
            if verbose:
                print_footer('Residual orthogonal to derivative.')
            return z / np.abs(z[-1])

        # def lsqr_D(z, dz, A, b, c, cache, residual):
        #     return residual_D(z, dz, A, b, c, cache)

        # def lsqr_DT(z, dres, A, b, c, cache, residual):
        #     return residual_DT(z, dres, A, b, c, cache)

        # print('res norm: %.2e' % np.linalg.norm(residual / z[-1]))
        # print('1 - btol: %.2e' % (1. - btol))
        returned = lsqr(A, b, c, cones, z,  # residual,
                        damp=0.,
                        atol=0.,  # max(10**(-1 - i), 1E-8),
                        btol=0.,  # max(10**(-1 - i), 1E-8),  # btol,
                        show=False,
                        iter_lim=lsqr_iters)  # None)  # )None)  # int(max_lsqr_iters))

        num_lsqr_iters = returned[2]
        step = returned[0]

        new_z = z - step
        # new_z /= np.abs(new_z[-1])
        new_residual, u, v, new_cache = residual_and_uv(new_z, A, b, c, cones)

        backtracks = 0
        # backtracking
        while np.linalg.norm(new_residual / new_z[-1]) > np.linalg.norm(residual / z[-1]):
            # print('backtracking')
            step /= 2.
            backtracks += 1
            new_z = z - step
            new_residual, u, v, new_cache = residual_and_uv(
                new_z, A, b, c, cones)

        # try one more divide by 2
        test_z = z - step / 2.
        test_residual, u, v, test_cache = residual_and_uv(
            test_z, A, b, c, cones)
        if np.linalg.norm(test_residual / test_z[-1]) < np.linalg.norm(new_residual / new_z[-1]):
            #print('swapping with shorter step')
            new_z = test_z
            new_residual = test_residual
            new_cache = test_cache
            backtracks += 1

            # end_z.append(np.copy(new_z))
        #end_z[:, i % mem_anderson] = new_z

        # enz_residual.append(np.copy(new_residual))

        if verbose:  # and (i % 10 == 0):
            print_stats(i + 1, new_residual, new_z,
                        # lambda dres: residual_DT(
                        #     new_z, dres, A, b, c, new_cache),
                        num_lsqr_iters, backtracks, start_time)
            # print('1 - btol: %.3f' % (1 - btol))

        rel_res_change = (np.linalg.norm(residual / z[-1]) - np.linalg.norm(
            new_residual / new_z[-1])) / np.linalg.norm(residual / z[-1])

        #print('rel residual change : %.2e' % rel_res_change)

        # if np.linalg.norm(new_residual / new_z[-1]) < np.linalg.norm(residual / z[-1]):
        #     # max(1. - (1. - btol) * 1.1, .5)
        #     btol = max(1. - rel_res_change, .1)
        # else:
        #     btol = 1. - (1. - btol) * 0.5

        # steps_anderson[:, i % mem_anderson_step] = new_z - z
        # arrivals_anderson[:, i % mem_anderson_step] = new_z

        old_z = np.copy(z)

        cache = new_cache
        z = new_z
        residual = new_residual

        if rel_res_change < 1E-8:
            if verbose:
                print_footer('Residual change too small.')
            return z / np.abs(z[-1])

        # if np.linalg.norm(new_residual / new_z[-1]) < np.linalg.norm(residual / z[-1]):
        #     cache = new_cache
        #     z = new_z
        #     residual = new_residual
        #     # max_lsqr_iters *= 1.1
        #     btol = max(1. - (1. - btol) * 2., .5)
        #     #btol = max(1. - rel_res_change, .5)
        #     # if (num_lsqr_iters < 2) and btol > 0.999:
        #     #     if verbose:
        #     #         print_footer("Expected residual change is too small.")
        #     #     return z / np.abs(z[-1])
        # else:
        #     btol = 1. - (1. - btol) * 0.5

        #     if btol > (1 - 1E-8):
        #         if verbose:
        #             print_stats(i + 1, residual, z,
        #                         num_lsqr_iters, start_time)
        #             print_footer("Residual change is too small.")

        #         return z / np.abs(z[-1])

            # if (num_lsqr_iters < 2):
            #     if verbose:
            #         print_footer("Can't make the residual smaller.")
            #     return z / np.abs(z[-1])
            # if verbose:
            #     print_footer("Can't make the residual smaller.")
            # return z / np.abs(z[-1])
            # max_lsqr_iters *= .5
            # btol = 1. - (1. - btol) * .5

        # if (time.time() - start_time) > max_runtime:
        #     if verbose:
        #         # print_stats(i + 1, residual, z,
        #         #             num_lsqr_iters, start_time)
        #         print_footer('Max. refinement runtime reached.')

        #     return z / np.abs(z[-1])

    if verbose:
        print_footer('Max num. refinement iters reached.')
    return z / np.abs(z[-1])


def solve(A, b, c, dim_dict,
          solver='scs',
          solver_options={},
          refine_solver_time_ratio=1.,
          max_iters=10,
          verbose=False,
          max_lsqr_iters=20,
          return_z=False):

    solver_start = time.time()
    if solver == 'scs':
        z, info = scs_solve(A, b, c, dim_dict, **solver_options)
    elif solver == 'ecos':
        z, info = ecos_solve(A, b, c, dim_dict, **solver_options)
    else:
        raise Exception('The only supported solvers are ecos and scs')

    solver_time = time.time() - solver_start

    cones = dim2cones(dim_dict)

    new_residual, u, v, _ = residual_and_uv(z, A, b, c, cones)
    x, s, y, tau, kappa = uv2xsytaukappa(u, v, A.shape[1])

    pres = np.linalg.norm(A@x + s - b) / (1 + np.linalg.norm(b))
    dres = np.linalg.norm(A.T@y + c) / (1 + np.linalg.norm(c))
    gap = np.abs(c@x + b@y) / (1 + np.abs(c@x) + np.abs(b@y))

    print('pres %.2e, dres %.2e, gap %.2e' % (pres, dres, gap))

    z_plus = refine(A, b, c, cones, z,
                    verbose=verbose,
                    iters=max_iters,
                    lsqr_iters=max_lsqr_iters)  # ,
    # max_runtime=solver_time * refine_solver_time_ratio)

    if return_z:
        return z_plus, info
    else:
        new_residual, u, v, _ = residual_and_uv(z_plus, A, b, c, cones)
        x, s, y, tau, kappa = uv2xsytaukappa(u, v, A.shape[1])
        pres = np.linalg.norm(A@x + s - b) / (1 + np.linalg.norm(b))
        dres = np.linalg.norm(A.T@y + c) / (1 + np.linalg.norm(c))
        gap = np.abs(c@x + b@y) / (1 + np.abs(c@x) + np.abs(b@y))
        print('pres %.2e, dres %.2e, gap %.2e' % (pres, dres, gap))
        return x, s, y, info
