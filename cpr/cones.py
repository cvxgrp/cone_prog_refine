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


----------------------------------------------------


Some parts (exponential cone projection) are adapted
from Brendan O'Donoghue's scs-python.
"""

import numpy as np
import scipy.sparse as sp
from collections import namedtuple

import numba as nb


class DimensionError(Exception):
    pass


class NonDifferentiable(Exception):
    pass


# def check_non_neg_int(n):
#     if not isinstance(n, int):
#         raise TypeError
#     if n < 0:
#         raise DimensionError


# def check_right_size(x, n):
#     if (n > 1) and not (len(x) == n):
#         raise DimensionError


cone = namedtuple('cone', ['Pi', 'D', 'DT'])


@nb.jit(nb.float64[:](nb.float64[:], nb.float64[:]))
def id_op(z, x):
    """Identity operator on x."""
    return np.copy(x)


@nb.jit(nb.float64[:](nb.float64[:]))
def free(z):
    """Projection on free cone, and cache."""
    return np.copy(z)


free_cone = cone(free, id_op, id_op)
free_cone_cached = cone(lambda z, cache: free_cone.Pi(z),
                        lambda z, x, cache: free_cone.D(z, x),
                        lambda z, x, cache: free_cone.D(z, x))


@nb.jit(nb.float64[:](nb.float64[:], nb.float64[:]))
def zero_D(z, x):
    """zero operator on x."""
    return np.zeros_like(x)


@nb.jit(nb.float64[:](nb.float64[:]))
def zero_Pi(z):
    """Projection on zero cone, and cache."""
    return np.zeros_like(z)


zero_cone = cone(zero_Pi, zero_D, zero_D)
zero_cone_cached = cone(lambda z, cache: zero_cone.Pi(z),
                        lambda z, x, cache: zero_cone.D(z, x),
                        lambda z, x, cache: zero_cone.D(z, x))


@nb.jit(nb.float64[:](nb.float64[:], nb.float64[:]), nopython=True)
def non_neg_D(z, x):
    # assert len(x) == len(z)
    result = np.copy(x)
    result[z < 0] = 0.
    return result


@nb.jit(nb.float64[:](nb.float64[:]), nopython=True)
def non_neg_Pi(z):
    """Projection on non-negative cone, and cache."""
    cache = np.zeros(1)
    return np.maximum(z, 0.)


non_neg_cone = cone(non_neg_Pi, non_neg_D, non_neg_D)
non_neg_cone_cached = cone(lambda z, cache: non_neg_cone.Pi(z),
                           lambda z, x, cache: non_neg_cone.D(z, x),
                           lambda z, x, cache: non_neg_cone.D(z, x))


# @nb.jit(nb.float64[:](nb.float64[:], nb.float64[:], nb.float64[:]), nopython=True)
# def sec_ord_D(z, x, cache):
#     """Derivative of projection on second order cone."""
#
#     rho, s = z[0], z[1:]
#     norm = cache[0]
#
#     if norm <= rho:
#         return np.copy(x)
#
#     if norm <= -rho:
#         return np.zeros_like(x)
#
#     y, t = x[1:], x[0]
#     normalized_s = s / norm
#     result = np.empty_like(x)
#     result[1:] = ((rho / norm + 1) * y -
#                   (rho / norm**3) * s *
#                   s@y +
#                   t * normalized_s) / 2.
#     result[0] = (y@ normalized_s + t) / 2.
#     return result

@nb.jit(nb.float64[:](nb.float64[:], nb.float64[:], nb.float64[:]))
def sec_ord_D(z, dz, cache):
    """Derivative of projection on second order cone."""

    # point at which we derive
    t = z[0]
    x = z[1:]
    # projection of point
    s = cache[0]
    y = cache[1:]

    # logic for simple cases
    norm = np.linalg.norm(x)
    if norm <= t:
        return np.copy(dz)
    if norm <= -t:
        return np.zeros_like(dz)

    # big case
    dx, dt = dz[1:], dz[0]

    # from my notes (attach photo)
    alpha = 2 * s - t
    if alpha == 0.:
        raise Exception('Second order cone derivative error')
    b = 2 * y - x
    c = dt * s + dx @ y
    d = dt * y + dx * s

    denom = (alpha - b @ b / alpha)
    if denom == 0.:
        raise Exception('Second order cone derivative error')
    ds = (c - b @ d / alpha) / denom
    dy = (d - b * ds) / alpha

    result = np.empty_like(dz)
    result[1:], result[0] = dy, ds
    return result

    # normalized_s = s / norm
    # result = np.empty_like(x)
    # result[1:] = ((rho / norm + 1) * y -
    #               (rho / norm**3) * s *
    #               s@y +
    #               t * normalized_s) / 2.
    # result[0] = (y@ normalized_s + t) / 2.
    # return result


@nb.jit(nb.float64[:](nb.float64[:], nb.float64[:]))
def sec_ord_Pi(z, cache):
    """Projection on second-order cone."""

    x, rho = z[1:], z[0]
    # cache this?
    norm_x = np.linalg.norm(x)
    # cache = np.zeros(1)
    # cache[0] = norm_x

    if norm_x <= rho:
        return np.copy(z)

    if norm_x <= -rho:
        return np.zeros_like(z)

    result = np.empty_like(z)

    result[0] = 1.
    result[1:] = x / norm_x
    result *= (norm_x + rho) / 2.

    cache[:] = result

    return result


sec_ord_cone = cone(sec_ord_Pi, sec_ord_D, sec_ord_D)


# @njit
# def _g_exp(z):
#     """Function used for exp. cone proj."""
#     r = z[0]
#     s = z[1]
#     t = z[2]
#     return s * np.exp(r / s) - t


# @njit
# def _gradient_g_exp(z):
#     """Gradient of function used for exp. cone proj."""
#     r = z[0]
#     s = z[1]
#     t = z[2]
#     result = np.empty(3)
#     result[0] = np.exp(r / s)
#     result[1] = result[0] * (1. - r / s)
#     result[2] = -1.
#     return result


# @njit
# def _hessian_g_exp(z):
#     """Hessian of function used for exp. cone proj."""
#     r = z[0]
#     s = z[1]
#     t = z[2]
#     result = np.zeros((3, 3))
#     ers = np.exp(r / s)
#     result[0, 0] = ers / s
#     result[0, 1] = - ers * r / s**2
#     result[1, 1] = ers * r**2 / s**3
#     result[1, 0] = result[0, 1]
#     return result


# @njit
# def _exp_residual(z_0, z, lambda_var):
#     """Residual of system for exp. cone projection."""
#     result = np.empty(4)
#     result[:3] = z - z_0 + lambda_var * _gradient_g_exp(z)
#     result[3] = _g_exp(z)
#     return result


# @njit
# def _exp_newt_matrix(z, lambda_var):
#     """Matrix used for exp. cone projection."""
#     result = np.empty((4, 4))
#     result[:3, :3] = lambda_var * _hessian_g_exp(z)
#     result[0, 0] += 1.
#     result[1, 1] += 1.
#     result[2, 2] += 1.
#     grad = _gradient_g_exp(z)
#     result[:3, 0] = grad
#     result[0, :3] = grad
#     result[3, 3] = 0.
#     return result

# def project_exp_bisection(v):
#   v = copy(v)
#   r = v[0]
#   s = v[1]
#   t = v[2]
#   # v in cl(Kexp)
#   if (s * exp(r / s) <= t and s > 0) or (r <= 0 and s == 0 and t >= 0):
#     return v
#   # -v in Kexp^*
#   if (-r < 0 and r * exp(s / r) <= -exp(1) * t) or (-r == 0 and -s >= 0 and
#                                                     -t >= 0):
#     return zeros(3,)
#   # special case with analytical solution
#   if r < 0 and s < 0:
#     v[1] = 0
#     v[2] = max(v[2], 0)
#     return v

#   x = copy(v)
#   ub, lb = get_rho_ub(v)
#   for iter in range(0, 100):
#     rho = (ub + lb) / 2
#     g, x = calc_grad(v, rho, x)
#     if g > 0:
#       lb = rho
#     else:
#       ub = rho
#     if ub - lb < 1e-6:
#       break
#   return x


@nb.jit(nb.float64(nb.float64, nb.float64, nb.float64, nb.float64))
def newton_exp_onz(rho, y_hat, z_hat, w):
    t = max(max(w - z_hat, -z_hat), 1e-6)
    for iter in np.arange(0, 100):
        f = (1 / rho**2) * t * (t + z_hat) - y_hat / rho + np.log(t / rho) + 1
        fp = (1 / rho**2) * (2 * t + z_hat) + 1 / t

        t = t - f / fp
        if t <= -z_hat:
            t = -z_hat
            break
        elif t <= 0:
            t = 0
            break
        elif np.abs(f) < 1e-6:
            break
    return t + z_hat


@nb.jit(nb.float64[:](nb.float64[:], nb.float64, nb.float64))
def solve_with_rho(v, rho, w):
    x = np.zeros(3)
    x[2] = newton_exp_onz(rho, v[1], v[2], w)
    x[1] = (1 / rho) * (x[2] - v[2]) * x[2]
    x[0] = v[0] - rho
    return x


@nb.jit(nb.types.Tuple((nb.float64, nb.float64[:]))(nb.float64[:],
                                                    nb.float64, nb.float64[:]),
        nopython=True)
def calc_grad(v, rho, warm_start):
    x = solve_with_rho(v, rho, warm_start[1])
    if x[1] == 0:
        g = x[0]
    else:
        g = (x[0] + x[1] * np.log(x[1] / x[2]))
    return g, x


@nb.jit(nb.types.UniTuple(nb.float64, 2)(nb.float64[:]))
def get_rho_ub(v):
    lb = 0
    rho = 2**(-3)
    g, z = calc_grad(v, rho, v)
    while g > 0:
        lb = rho
        rho = rho * 2
        g, z = calc_grad(v, rho, z)
    ub = rho
    return ub, lb


@nb.jit(nb.types.UniTuple(nb.float64[:], 2)(nb.float64[:]))
def fourth_case_brendan(z):
    x = np.copy(z)
    ub, lb = get_rho_ub(x)
    # print('initial ub, lb', ub, lb)
    for i in range(0, 100):
        rho = (ub + lb) / 2
        g, x = calc_grad(z, rho, x)
        # print('g, x', g, x)
        if g > 0:
            lb = rho
        else:
            ub = rho
        # print('new ub, lb', ub, lb)
        if ub - lb < 1e-8:
            break
    # print('result:', x)
    return x, np.copy(x)


# Here it is my work on exp proj D
@nb.jit(nb.float64[:, :](nb.float64, nb.float64, nb.float64, nb.float64,
                         nb.float64, nb.float64))
def make_mat(r, s, t, x, y, z):
    mat = np.zeros((3, 3))
    # first eq.
    mat[0, 0] = 2 * x - r
    mat[0, 1] = 2 * y - s
    mat[0, 2] = 2 * z - t
    mat[1, 0] = np.exp(x / y)
    mat[1, 1] = np.exp(x / y) * (1 - x / y)
    mat[1, 2] = -1.
    mat[2, 0] = y
    mat[2, 1] = x - r
    mat[2, 2] = 2 * z - t
    return mat

# Here it is my work on exp proj D


@nb.jit(nb.float64[:, :](nb.float64, nb.float64, nb.float64, nb.float64,
                         nb.float64, nb.float64))
def make_mat_two(r, s, t, x, y, z):
    mat = np.zeros((3, 3))
    # first eq.
    u = -(r - x)
    v = -(s - y)
    w = -(t - z)

    mat[0, 0] = 2 * x - r
    mat[0, 1] = 2 * y - s
    mat[0, 2] = 2 * z - t

    # mat[1, 0] = np.exp(x / y)
    # mat[1, 1] = np.exp(x / y) * (1 - x / y)
    # mat[1, 2] = -1.

    mat[1, 0] = np.exp(v / u) * (1 - v / u)
    mat[1, 1] = np.exp(v / u)
    mat[1, 2] = np.e

    mat[2, 0] = y
    mat[2, 1] = x - r
    mat[2, 2] = 2 * z - t
    return mat


@nb.jit(nb.float64[:](nb.float64, nb.float64, nb.float64, nb.float64,
                      nb.float64, nb.float64))
def make_error(r, s, t, x, y, z):
    error = np.zeros(3)
    error[0] = x * (x - r) + y * (y - s) + z * (z - t)
    error[1] = y * np.exp(x / y) - z
    error[2] = y * (x - r) + z * (z - t)
    # print('error', error)
    return error


@nb.jit(nb.float64[:](nb.float64, nb.float64, nb.float64, nb.float64,
                      nb.float64, nb.float64))
def make_error_two(r, s, t, x, y, z):
    error = np.zeros(3)
    error[0] = x * (x - r) + y * (y - s) + z * (z - t)
    u = -(r - x)
    v = -(s - y)
    w = -(t - z)
    error[1] = np.e * w + u * np.exp(v / u)
    error[2] = y * (x - r) + z * (z - t)
    # print('error', error)
    return error


@nb.jit(nb.float64[:](nb.float64, nb.float64, nb.float64, nb.float64,
                      nb.float64, nb.float64))
def make_rhs(x, y, z, dr, ds, dt):
    rhs = np.zeros(3)
    rhs[0] = x * dr + y * ds + z * dt
    rhs[2] = y * dr + z * dt
    return rhs


@nb.jit(nb.float64[:](nb.float64, nb.float64, nb.float64, nb.float64,
                      nb.float64, nb.float64, nb.float64, nb.float64, nb.float64))
def fourth_case_D(r, s, t, x, y, z, dr, ds, dt):

    test = np.zeros(3)
    test[0], test[1], test[2] = r, s, t

    u = x - r

    if y < 1E-12:
        return np.zeros(3)

    # if y > -u and not y == 0.:  # temporary fix?
        # print('computing error with e^(x/y)')
    error = make_error(r, s, t, x, y, z)
    # else:
    # print('computing error with e^(v/u)')
    #    error = make_error_two(r, s, t, x, y, z)

    # error = make_error(r, s, t, x, y, z)
    rhs = make_rhs(x, y, z, dr, ds, dt)
    # print('base rhs', rhs)
    if np.any(np.isnan(error)) or np.any(np.isinf(error)) or np.any(np.isnan(rhs)):
        return np.zeros(3)

    # if y > -u and not y == 0.:  # temporary fix?
        # print('solving system with e^(x/y)')
    result = np.linalg.solve(make_mat(r, s, t, x, y, z) + np.eye(3) * 1E-8,
                             rhs - error)
    # else:
    # print('solving system with e^(v/u)')
    #    result = np.linalg.solve(make_mat_two(r, s, t, x, y, z),  # + np.eye(3) * 1E-8,
    #                             rhs - error)

    if np.any(np.isnan(result)):
        # raise Exception('Exp cone derivative error.')
        return np.zeros(3)
    # print('result', result)
    return result


@nb.jit(nb.float64[:](nb.float64[:]))
def fourth_case_enzo(z_var):
    # VERBOSE = False

    # print('fourth case Enzo')
    # if VERBOSE:
    #    print('projecting (%f, %f, %f)' % (z_var[0], z_var[1], z_var[2]))
    real_result, _ = fourth_case_brendan(z_var)
    # print('brendan result: (x,y,z)', real_result)
    z_var = np.copy(z_var)
    r = z_var[0]
    s = z_var[1]
    t = z_var[2]
    # print('brendan result: (u,v,w)', real_result - z_var)

    x, y, z = real_result
    # if y < 1E-12:
    #     y = 1E-12

    for i in range(10):

        # if y < 1e-14:
        #     return np.zeros(3)
        u = x - r
        # assert y >= 0
        # assert u <= 0
        if (y <= 0. and u >= 0.):
            return np.zeros(3)

        if y > -u and not y == 0.:
            # print('computing error with e^(x/y)')
            error = make_error(r, s, t, x, y, z)
        else:
            # print('computing error with e^(v/u)')
            if u == 0:
                break
            error = make_error_two(r, s, t, x, y, z)

        # print('error:', error)
        # print('error norm: %.2e' % np.linalg.norm(error))

        # if VERBOSE:
        # print('iter %d, max |error| = %g' % (i, np.max(np.abs(error))))

        if np.max(np.abs(error)) < 1e-15:
            # print(np.max(np.abs(error)))
            # print('converged!')
            break

        if np.any(np.isnan(error)):
            break

        if y > -u and not y == 0.:
            # print('computing correction with e^(x/y)')
            correction = np.linalg.solve(
                make_mat(r, s, t, x, y, z) + np.eye(3) * 1E-8,
                -error)
        else:
            # print('computing correction with e^(v/u)')
            correction = np.linalg.solve(
                make_mat_two(r, s, t, x, y, z) + np.eye(3) * 1E-8,
                -error)

        # print('correction', correction)

        if np.any(np.isnan(correction)):
            break

        x += correction[0]
        y += correction[1]
        z += correction[2]

    result = np.empty(3)
    result[0] = x
    result[1] = y
    result[2] = z

    # if VERBOSE:
    #    print('result = (%f, %f, %f)' % (result[0], result[1], result[2]))

    return result


@nb.jit(nb.float64[:](nb.float64[:], nb.float64[:]), nopython=True)
def exp_pri_Pi(z, cache):
    """Projection on exp. primal cone, and cache."""
    z = np.copy(z)
    r = z[0]
    s = z[1]
    t = z[2]

    # temporary...
    if np.linalg.norm(z) < 1E-14:
        cache[:3] = 0.
        return z

    # first case
    if (s > 0 and s * np.exp(r / s) <= t) or \
            (r <= 0 and s == 0 and t >= 0):
        cache[:3] = z
        return z

    # second case
    if (-r < 0 and r * np.exp(s / r) <= -np.exp(1) * t) or \
            (r == 0 and -s >= 0 and -t >= 0):
        cache[:3] = 0.
        return np.zeros(3)

    # third case
    if r < 0 and s < 0:
        z[1] = 0
        z[2] = max(z[2], 0)
        cache[:3] = z
        return z

    pi = fourth_case_enzo(z)
    cache[:3] = pi
    return pi


@nb.jit(nb.float64[:](nb.float64[:], nb.float64[:], nb.float64[:]), nopython=True)
def exp_pri_D(z_0, dz, cache):
    """Derivative of proj. on exp. primal cone."""

    r = z_0[0]
    s = z_0[1]
    t = z_0[2]

    dr = dz[0]
    ds = dz[1]
    dt = dz[2]

    # projection of z_0
    x = cache[0]
    y = cache[1]
    z = cache[2]

    # first case
    # if on the cone boundary, non-diff
    if (s > 0 and s * np.exp(r / s) == t) or \
            (r <= 0 and s == 0 and t >= 0):  # or \
            # (r <= 0 and s == 0 and t == 0):
        # raise NonDifferentiable
        return np.zeros(3)

    if (s > 0 and s * np.exp(r / s) < t):
        # print('first case')
        return np.copy(dz)

    # second case
    # if on cone bound, then non-diff
    if (-r < 0 and r * np.exp(s / r) == -np.exp(1) * t) or \
            (r == 0 and -s >= 0 and -t >= 0):  # or \
        # (r == 0 and -s >= 0 and -t == 0):
        # raise NonDifferentiable
        return np.zeros(3)

    if (-r < 0 and r * np.exp(s / r) < -np.exp(1) * t):  # or \
           # (r == 0 and -s > 0 and -t > 0):
        # print('second case')
        return np.zeros(3)

    if r < 0 and s < 0 and t == 0:
        # raise NonDifferentiable
        return np.zeros(3)

    # third case
    if r < 0 and s < 0:
        # print('third case')
        result = np.zeros(3)
        result[0] = dz[0]
        result[2] = dz[2] if t > 0 else 0.
        # print('result', result)
        return result

    # fourth case
    fourth = fourth_case_D(r, s, t, x, y, z, dr, ds, dt)
    # assert not True in np.isnan(fourth)
    return fourth


exp_pri_cone = cone(exp_pri_Pi, exp_pri_D, exp_pri_D)


@nb.jit(nb.float64[:](nb.float64[:], nb.float64[:]))
def exp_dua_Pi(z, cache):
    """Projection on exp. dual cone, and cache."""
    minuspi = exp_pri_Pi(-z, cache)
    return np.copy(z) + minuspi


@nb.jit(nb.float64[:](nb.float64[:], nb.float64[:], nb.float64[:]))
def exp_dua_D(z, x, cache):
    """Derivative of projection on exp. dual cone."""
    # res = exp_pri_D(z, x, cache)
    return np.copy(x) + exp_pri_D(-z, -x, cache)


exp_dua_cone = cone(exp_dua_Pi, exp_dua_D, exp_dua_D)


@nb.jit(nb.int64(nb.int64))
def sizevec2sizemat(n):
    m = int(np.sqrt(2 * n))
    if not n == (m * (m + 1) / 2.):
        raise DimensionError
    return m


@nb.jit(nb.int64(nb.int64))
def sizemat2sizevec(m):
    return int((m * (m + 1) / 2.))


@nb.jit(nb.float64[:](nb.float64[:, :]))
def mat2vec(Z):
    """Upper tri row stacked, off diagonal multiplied by sqrt(2)."""
    n = Z.shape[0]
    l = sizemat2sizevec(n)
    result = np.empty(l)
    cur = 0
    sqrt_two = np.sqrt(2.)

    for i in range(n):  # row
        result[cur] = Z[i, i]
        cur += 1
        for j in range(i + 1, n):  # column
            result[cur] = Z[i, j] * sqrt_two
            cur += 1

    return result


@nb.jit(nb.float64[:, :](nb.float64[:]))
def vec2mat(z):
    n = sizevec2sizemat(len(z))

    cur = 0
    result = np.empty((n, n))
    sqrt_two = np.sqrt(2)

    for i in range(n):  # row
        result[i, i] = z[cur]
        cur += 1
        for j in range(i + 1, n):  # column
            result[i, j] = z[cur] / sqrt_two
            result[j, i] = result[i, j]
            cur += 1

    return result


# @nb.jit()  # nb.float64[:](nb.float64[:, :]))
# def flatten(mat):
#     return mat.flatten()


@nb.jit(nb.float64[:, :](nb.float64[:]), nopython=True)
def reconstruct_sqmat(vec):
    n = int(np.sqrt(len(vec)))
    return np.copy(vec).reshape((n, n))  # copying because of numba issue


@nb.jit((nb.int64, nb.float64[:], nb.float64[:], nb.float64[:, :],
         nb.float64[:, :]), nopython=True)
def inner(m, lambda_plus, lambda_minus, dS_tilde, dZ_tilde):
    for i in range(m):
        for j in range(i, m):
            nom = lambda_plus[j]
            denom = lambda_plus[j] + lambda_minus[i]
            if (nom == 0.) and (denom == 0.):
                factor = 0.
            else:
                factor = nom / denom
            dS_tilde[i, j] = dZ_tilde[i, j] * factor
            dS_tilde[j, i] = dS_tilde[i, j]


# nb.float64[:](nb.float64[:], nb.float64[:], nb.float64[:],
@nb.jit(nb.float64[:](nb.float64[:],
                      nb.float64[:],
                      nb.float64[:],
                      nb.float64[:]), nopython=True)
def semidef_cone_D(z, dz, cache_eivec, cache_eival):
    U, lambda_var = reconstruct_sqmat(cache_eivec), cache_eival
    dZ = vec2mat(dz)
    dZ_tilde = U.T @ dZ @ U
    lambda_plus = np.maximum(lambda_var, 0.)
    lambda_minus = -np.minimum(lambda_var, 0.)
    m = len(lambda_plus)
    k = np.sum(lambda_plus > 0)
    # if not (np.sum(lambda_minus > 0) + k == m):
    #   raise NonDifferentiable('SD cone projection not differentiable!')
    dS_tilde = np.empty((m, m))
    inner(m, lambda_plus, lambda_minus, dS_tilde, dZ_tilde)
    dS = U @ dS_tilde @ U.T
    return mat2vec(dS)


@nb.jit(nb.float64[:](nb.float64[:], nb.float64[:],
                      nb.float64[:]), nopython=True)
def semidef_cone_Pi(z, cache_eivec, cache_eival):

    Z = vec2mat(z)
    eival, eivec = np.linalg.eigh(Z)
    result = mat2vec(eivec @ np.diag(np.maximum(eival, 0.)) @ eivec.T)

    Pi_Z = eivec @ np.diag(np.maximum(eival, 0.)) @ eivec.T

    # print('semidef proj Z = %s' % z)
    # Pi_star_Z = eivec @ np.diag(np.minimum(eival, 0.)) @ eivec.T
    # error = Pi_Z @ Pi_star_Z
    # print('eival = %s' % eival)
    # print('semidef_proj_error: %.2e' % np.linalg.norm(error))

    cache_eivec[:] = eivec.flatten()
    cache_eival[:] = eival

    return result


semi_def_cone = cone(semidef_cone_Pi, semidef_cone_D, semidef_cone_D)


@nb.jit(nb.float64[:](nb.float64[:], nb.float64[:],
                      nb.types.Tuple((nb.float64[:], nb.float64[:]))))
def semidef_cone_D_single_cache(z, dz, cache):
    return semidef_cone_D(z, dz, cache[0], cache[1])


@nb.jit(nb.float64[:](nb.float64[:],
                      nb.types.Tuple((nb.float64[:], nb.float64[:]))))
def semidef_cone_Pi_single_cache(z, cache):
    return semidef_cone_Pi(z, cache[0], cache[1])


# used as test harness for semi-def functions
semi_def_cone_single_cache = cone(semidef_cone_Pi_single_cache, semidef_cone_D_single_cache,
                                  semidef_cone_D_single_cache)

# these functions are used to store matrices in cache


cache_types = nb.types.Tuple((nb.int64, nb.int64,  # z, l
                              nb.int64[:],
                              nb.float64[:],
                              # nb.types.List(nb.float64[:]),  # q
                              nb.int64[:],  # s
                              # nb.types.List(nb.float64[:, :]),
                              nb.float64[:],
                              # nb.types.List(nb.float64[:]),
                              nb.float64[:],
                              nb.int64, nb.float64[:, :],  # ep
                              nb.int64, nb.float64[:, :]))  # ed


# nb.float64[:](nb.float64[:], nb.float64[:], nb.int64, nb.int64,  # z, l
#                       nb.int64[:],
#                       nb.float64[:],
#                       # nb.types.List(nb.float64[:]),  # q
#                       nb.int64[:],  # s
#                       #nb.types.List(nb.float64[:, :]),
#                       nb.float64[:],
#                       # nb.types.List(nb.float64[:]),
#                       nb.float64[:],
#                       nb.int64,
#                       nb.float64[:, :],  # ep
#                       nb.int64,
#                       nb.float64[:, :]), nopython=True)
@nb.jit(nopython=True)
def prod_cone_D(z, x, zero, l, q, q_cache, s, s_cache_eivec, s_cache_eival, ep, ep_cache, ed, ed_cache):

    result = np.empty_like(z)
    cur = 0

    # zero
    result[cur:cur + zero] = zero_D(z[cur:cur + zero], x[cur:cur + zero])
    cur += zero

    # non-neg
    result[cur:cur + l] = non_neg_D(z[cur:cur + l], x[cur:cur + l])
    cur += l

    # second order
    sec_ord_cache_cur = 0
    for index, size in enumerate(q):
        result[cur:cur + size] = sec_ord_D(z[cur:cur + size],
                                           x[cur:cur + size],
                                           q_cache[sec_ord_cache_cur:
                                                   sec_ord_cache_cur + size])
        cur += size
        sec_ord_cache_cur += size

    # semi-def
    semi_def_cache_cur = 0
    semi_def_eivec_cache_cur = 0

    for index, size in enumerate(s):
        vecsize = sizemat2sizevec(size)
        result[cur:cur + vecsize] = semidef_cone_D(z[cur:cur + vecsize],
                                                   x[cur:cur + vecsize],
                                                   s_cache_eivec[
            semi_def_eivec_cache_cur:semi_def_eivec_cache_cur + size**2],
            s_cache_eival[semi_def_cache_cur:semi_def_cache_cur + size])

        cur += vecsize
        semi_def_cache_cur += size
        semi_def_eivec_cache_cur += size**2

    # exp-pri
    for index in range(ep):
        result[cur:cur + 3] = exp_pri_D(z[cur:cur + 3],
                                        x[cur:cur + 3],
                                        ep_cache[index])
        cur += 3

    # exp-dua
    for index in range(ed):
        result[cur:cur + 3] = exp_dua_D(z[cur:cur + 3],
                                        x[cur:cur + 3],
                                        ed_cache[index])
        cur += 3

    assert cur == len(z)
    return result


# @nb.jit(nb.float64[:](nb.float64[:], nb.int64, nb.int64,  # z, l
#                       nb.int64[:],
#                       nb.float64[:],
#                       # nb.types.List(nb.float64[:]),  # q
#                       nb.int64[:],  # s
#                       #nb.types.List(nb.float64[:, :]),
#                       nb.float64[:],
#                       # nb.types.List(nb.float64[:]),
#                       nb.float64[:],
#                       nb.int64, nb.float64[:, :],  # ep
#                       nb.int64, nb.float64[:, :]),
@nb.jit(nopython=True)
def prod_cone_Pi(z, zero, l, q, q_cache, s, s_cache_eivec, s_cache_eival,  ep,
                 ep_cache, ed, ed_cache):
    """Projection on product of zero, non-neg, sec. ord., sd, exp p., exp d."""

    result = np.empty_like(z)
    cur = 0

    # zero
    result[cur:cur + zero] = zero_Pi(z[cur:cur + zero])
    cur += zero

    # non-neg
    result[cur:cur + l] = non_neg_Pi(z[cur:cur + l])
    cur += l

    # second order
    sec_ord_cache_cur = 0
    for index, size in enumerate(q):

        result[cur:cur + size] = sec_ord_Pi(z[cur:cur + size],
                                            q_cache[sec_ord_cache_cur:
                                                    sec_ord_cache_cur + size])
        cur += size
        sec_ord_cache_cur += size

    # semi-def
    semi_def_cache_cur = 0
    semi_def_eivec_cache_cur = 0

    for index, size in enumerate(s):
        vecsize = sizemat2sizevec(size)
        result[cur:cur + vecsize] = semidef_cone_Pi(z[cur:cur + vecsize],
                                                    s_cache_eivec[semi_def_eivec_cache_cur:
                                                                  semi_def_eivec_cache_cur + size**2],
                                                    s_cache_eival[semi_def_cache_cur:
                                                                  semi_def_cache_cur + size])
        cur += vecsize
        semi_def_cache_cur += size
        semi_def_eivec_cache_cur += size**2

    # exp-pri
    for index in range(ep):
        result[cur:cur + 3] = exp_pri_Pi(z[cur:cur + 3],
                                         ep_cache[index])
        cur += 3

    # exp-dua
    for index in range(ed):
        result[cur:cur + 3] = exp_dua_Pi(z[cur:cur + 3],
                                         ed_cache[index])
        cur += 3

    return result


prod_cone = cone(prod_cone_Pi, prod_cone_D, prod_cone_D)


def make_prod_cone_cache(dim_dict):
    return _make_prod_cone_cache(dim_dict['z'] if 'z' in dim_dict else 0,
                                 dim_dict['l'] if 'l' in dim_dict else 0,
                                 np.array(dim_dict['q'] if 'q'
                                          in dim_dict else [], dtype=np.int64),
                                 np.array(dim_dict['s'] if 's' in
                                          dim_dict else [], dtype=np.int64),
                                 dim_dict['ep'] if 'ep' in dim_dict else 0,
                                 dim_dict['ed'] if 'ed' in dim_dict else 0,)


# @nb.jit(cache_types(nb.int64, nb.int64, nb.int64[:], nb.int64[:],
#                     nb.int64, nb.int64), nopython=True)
@nb.jit(nopython=True)
def _make_prod_cone_cache(zero, non_neg, second_ord, semi_def, exp_pri, exp_dua):

    # q_cache = []
    # for size in second_ord:
    #     q_cache.append(np.zeros(size))

    q_cache = np.zeros(np.sum(second_ord))

    # q_cache.append(np.zeros(1))  # for numba

    # s_cache_eivec = []
    s_cache_eival = np.zeros(np.sum(semi_def))
    s_cache_eivec = np.zeros(np.sum(semi_def**2))
    # for matrix_size in semi_def:
    #     s_cache_eivec.append(np.zeros((matrix_size, matrix_size)))
    # s_cache_eival.append(np.zeros(matrix_size))

    # s_cache_eivec.append(np.zeros((1, 1)))
    # s_cache_eival.append(np.zeros(1))

    return zero, non_neg, second_ord, q_cache, \
        semi_def, s_cache_eivec, s_cache_eival, \
        exp_pri, np.zeros((exp_pri, 3)), \
        exp_dua, np.zeros((exp_dua, 3))


# @nb.jit(nb.float64[:](nb.float64[:], nb.float64[:], cache_types, nb.int64),
#         nopython=True)
#@nb.jit()  # nopython=True)
# @nb.jit(nb.float64[:](nb.float64[:], nb.float64[:],
#                      nb.int64, nb.int64,  # z, l
#                      nb.int64[:], nb.types.List(nb.float64[:]),  # q
#                      nb.int64[:],  # s
#                      nb.types.List(nb.float64[:, :]),
#                      nb.types.List(nb.float64[:]),
#                      nb.int64, nb.float64[:, :],  # ep
#                      nb.int64, nb.float64[:, :],
#                      nb.int64), nopython=True)
@nb.jit(nopython=True)
def embedded_cone_D(z, dz, zero, l, q, q_cache, s, s_cache_eivec,
                    s_cache_eival,  ep,
                    ep_cache, ed, ed_cache, n):
    """Der. of proj. on the cone of the primal-dual embedding."""
    # return dz - prod_cone.D(-z, dz, cache)
    result = np.empty_like(dz)

    result[:n] = dz[:n]

    # ds = prod_cone_D(-z[n:-1], dz[n:-1], zero, l, q, q_cache, s, s_cache_eivec,
    #                  s_cache_eival, ep, ep_cache, ed, ed_cache)
    result[n:-1] = dz[n:-1] - prod_cone_D(-z[n:-1], dz[n:-1], zero, l, q,
                                          q_cache, s, s_cache_eivec,
                                          s_cache_eival, ep, ep_cache,
                                          ed, ed_cache)
    result[-1:] = non_neg_D(z[-1:], dz[-1:])

    return result


# @nb.jit(nb.float64[:](nb.float64[:], cache_types, nb.int64),
#         nopython=True)
# @nb.jit(nopython=True)
# @nb.jit(nb.float64[:](nb.float64[:],
#                      nb.int64, nb.int64,  # z, l
#                      nb.int64[:], nb.types.List(nb.float64[:]),  # q
#                      nb.int64[:],  # s
#                      nb.types.List(nb.float64[:, :]),
#                      nb.types.List(nb.float64[:]),
#                      nb.int64, nb.float64[:, :],  # ep
#                      nb.int64, nb.float64[:, :],
#                      nb.int64), nopython=True)
@nb.jit(nopython=True)
def embedded_cone_Pi(z, zero, l, q, q_cache, s, s_cache_eivec, s_cache_eival,
                     ep, ep_cache, ed, ed_cache, n):
    """Projection on the cone of the primal-dual embedding."""

    # emb_cones = [[zero_cone, n]] + cones + [[non_neg_cone, 1]]
    # v, cache = prod_cone.Pi(-z, emb_cones)
    # return v + z, cache

    result = np.empty_like(z)
    result[:n] = z[:n]  # x
    result[n:-1] = z[n:-1] + prod_cone_Pi(-z[n:-1], zero, l, q, q_cache, s,
                                          s_cache_eivec,
                                          s_cache_eival, ep, ep_cache,
                                          ed, ed_cache)
    result[-1:] = non_neg_Pi(z[-1:])  # tau
    return result


embedded_cone = cone(embedded_cone_Pi, embedded_cone_D, embedded_cone_D)
