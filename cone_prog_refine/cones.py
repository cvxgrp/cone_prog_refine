"""
Enzo Busseti, Walaa Moursi, Stephen Boyd, 2018.

Some parts (exponential cone projection) are adapted
from Brendan O'Donoghue's scs-python.

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
import scipy.sparse as sp
from collections import namedtuple
from numba import njit, jit
from functools import lru_cache


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


@njit
def id_op(z, x, cache):
    """Identity operator on x."""
    return np.copy(x)


@njit
def free(z):
    """Projection on free cone, and cache."""
    cache = None
    return np.copy(z), cache


free_cone = cone(free, id_op, id_op)


@njit
def zero_D(z, x, cache):
    """zero operator on x."""
    return np.zeros_like(x)


@njit
def zero_Pi(z):
    """Projection on zero cone, and cache."""
    cache = None
    return np.zeros_like(z), cache

zero_cone = cone(zero_Pi, zero_D, zero_D)


@njit
def non_neg_D(z, x, cache=None):
    assert len(x) == len(z)
    result = np.copy(x)
    result[z < 0] = 0.
    return result


@njit
def non_neg_Pi(z):
    """Projection on non-negative cone, and cache."""
    cache = None
    return np.maximum(z, 0.), cache


non_neg_cone = cone(non_neg_Pi, non_neg_D, non_neg_D)


@njit
def sec_ord_D(z, x, cache):
    """Derivative of projection on second order cone."""

    rho, s = z[0], z[1:]
    norm = cache

    if norm <= rho:
        return np.copy(x)

    if norm <= -rho:
        return np.zeros_like(x)

    y, t = x[1:], x[0]
    normalized_s = s / norm
    result = np.empty_like(x)
    result[1:] = ((rho / norm + 1) * y -
                  (rho / norm**3) * s *
                  s@y +
                  t * normalized_s) / 2.
    result[0] = (y@ normalized_s + t) / 2.
    return result


@njit
def sec_ord_Pi(z):
    """Projection on second-order cone."""

    x, rho = z[1:], z[0]
    # cache this?
    norm_x = np.linalg.norm(x)
    cache = norm_x

    if norm_x <= rho:
        return np.copy(z), cache

    if norm_x <= -rho:
        return np.zeros_like(z), cache

    result = np.empty_like(z)

    result[0] = 1.
    result[1:] = x / norm_x
    result *= (norm_x + rho) / 2.

    return result, cache


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


@njit
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


@njit
def solve_with_rho(v, rho, w):
    x = np.zeros(3)
    x[2] = newton_exp_onz(rho, v[1], v[2], w)
    x[1] = (1 / rho) * (x[2] - v[2]) * x[2]
    x[0] = v[0] - rho
    return x


@njit
def calc_grad(v, rho, warm_start):
    x = solve_with_rho(v, rho, warm_start[1])
    if x[1] == 0:
        g = x[0]
    else:
        g = (x[0] + x[1] * np.log(x[1] / x[2]))
    return g, x


@njit
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


@njit
def fourth_case_brendan(z):
    x = np.copy(z)
    ub, lb = get_rho_ub(x)
    #print('initial ub, lb', ub, lb)
    for i in range(0, 100):
        rho = (ub + lb) / 2
        g, x = calc_grad(z, rho, x)
        #print('g, x', g, x)
        if g > 0:
            lb = rho
        else:
            ub = rho
        #print('new ub, lb', ub, lb)
        if ub - lb < 1e-8:
            break
    #print('result:', x)
    return x, np.copy(x)


# Here it is my work on exp proj D
@njit
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


@njit
def make_error(r, s, t, x, y, z):
    error = np.zeros(3)
    error[0] = x * (x - r) + y * (y - s) + z * (z - t)
    error[1] = y * np.exp(x / y) - z
    error[2] = y * (x - r) + z * (z - t)
    #print('error', error)
    return error


@njit
def make_rhs(x, y, z, dr, ds, dt):
    rhs = np.zeros(3)
    rhs[0] = x * dr + y * ds + z * dt
    rhs[2] = y * dr + z * dt
    return rhs


@njit
def fourth_case_D(r, s, t, x, y, z, dr, ds, dt):

    #print('fourth case D')

    error = make_error(r, s, t, x, y, z)
    rhs = make_rhs(x, y, z, dr, ds, dt)
    #print('base rhs', rhs)

    result = np.linalg.solve(make_mat(r, s, t, x, y, z),
                             rhs - error)
    #print('result', result)
    return result


@njit
def fourth_case_enzo(z_var):

    real_result, _ = fourth_case_brendan(z_var)
    z_var = np.copy(z_var)
    r = z_var[0]
    s = z_var[1]
    t = z_var[2]

    #print('fourth case enzo', z)

    #print('real_result', real_result)

    # if s > 0:
    #     x = r
    #     y = s
    #     z = s * np.exp(r / s)

    # else:
    #     assert r > 0.
    #     assert t > 0.
    #     x = r
    #     y = 1.
    #     z = y * np.exp(x / y)

    x, y, z = real_result
    if y == 0.:
        y = 1E-8

    #print('candidate:', (x, y, z))

    for i in range(10):

        error = make_error(r, s, t, x, y, z)

        #print('it %d, error norm %.2e' % (i, np.linalg.norm(error)))

        if np.linalg.norm(error) < 1E-14:
            break

        correction = np.linalg.solve(
            make_mat(r, s, t, x, y, z),
            -error)

        if np.linalg.norm(correction) < 1E-12:
            break

        #print('correction', correction)

        step = 1.

        while True:

            new_x = x + step * correction[0]
            new_y = y + step * correction[1]
            new_z = z + step * correction[2]
            new_error = make_error(r, s, t, new_x, new_y, new_z)

            if (np.linalg.norm(new_error) <= np.linalg.norm(error)) and \
                    new_y > 0:
                # accept
                x = new_x
                y = new_y
                z = new_z
                break

            step /= 2.
        #print('final step', step)

        #print('current:', (x, y, z))

    #print('result:', np.array([x, y, z]))

    result = np.empty(3)
    result[0] = x
    result[1] = y
    result[2] = z

    return result, np.copy(result)


#@njit

@njit
def exp_pri_Pi(z):
    #print('\nprojecting %s' % z)
    """Projection on exp. primal cone, and cache."""
    z = np.copy(z)
    # cache = None
    r = z[0]
    s = z[1]
    t = z[2]

    # first case
    if (s > 0 and s * np.exp(r / s) <= t) or \
            (r <= 0 and s == 0 and t >= 0):
        #print('first case')
        return z, np.copy(z)

    # second case
    if (-r < 0 and r * np.exp(s / r) <= -np.exp(1) * t) or \
            (r == 0 and -s >= 0 and -t >= 0):
        #print('second case')
        return np.zeros(3), np.zeros(3)

    # third case
    if r < 0 and s < 0:
        #print('third case')
        z[1] = 0
        z[2] = max(z[2], 0)
        return z, np.copy(z)

    # fourth case
    # z_proj = np.copy(z)
    # lambda_var = 1.
    # residual = _exp_residual(z, z_proj, lambda_var)
    # resnorm = np.linalg.norm(residual)
    # print('init. residual', residual)
    # print('init resnorm', resnorm)
    # j = 0.
    # while not np.allclose(0., residual, atol=1E-6):
    #     step = np.linalg.solve(
    #         _exp_newt_matrix(z_proj, lambda_var),
    #         residual)
    #     print('\nstep', step)

    #     alpha = 1.
    #     old_proj = np.copy(z_proj)
    #     old_lambda = np.copy(lambda_var)
    #     old_res_norm = np.copy(resnorm)
    #     j += 1
    #     if j > 100:
    #         break
    #     i = 0.
    #     while True:
    #         print('alpha', alpha)
    #         z_proj = old_proj - alpha * step[:3]
    #         lambda_var = old_lambda - alpha * step[3]
    #         residual = _exp_residual(z, z_proj, lambda_var)
    #         resnorm = np.linalg.norm(residual)
    #         print('new_residual btrk', residual)
    #         print('new_res_norm', resnorm)
    #         print('oldres_norm', old_res_norm)
    #         if (resnorm < old_res_norm) \
    #                 and z_proj[1] > 0.:
    #             break
    #         alpha /= 2.
    #         i += 1
    #         if i > 100:
    #             break

    #print('fourth case')
    # return fourth_case_brendan(z)
    return fourth_case_enzo(z)


@njit
def exp_pri_D(z_0, dz, cache):
    """Derivative of proj. on exp. primal cone."""
    # z = np.copy(z)
    # cache = None
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
    # print('\nrst', r, s, t)
    # print('dr ds dt', dr, ds, dt)
    # print('xyz', x, y, z)

    # first case
    # if on the cone boundary, non-diff
    if (s > 0 and s * np.exp(r / s) == t) or \
            (r <= 0 and s == 0 and t >= 0):  # or \
            #(r <= 0 and s == 0 and t == 0):
        raise NonDifferentiable

    if (s > 0 and s * np.exp(r / s) < t):
        #print('first case')
        return np.copy(dz)

    # if (r < 0 and s == 0 and t > 0):
    #     # i think no
    #     return np.copy(dz)

    # second case
    # if on cone bound, then non-diff
    if (-r < 0 and r * np.exp(s / r) == -np.exp(1) * t) or \
            (r == 0 and -s >= 0 and -t >= 0):  # or \
        #(r == 0 and -s >= 0 and -t == 0):
        raise NonDifferentiable

    if (-r < 0 and r * np.exp(s / r) < -np.exp(1) * t):  # or \
           # (r == 0 and -s > 0 and -t > 0):
        #print('second case')
        return np.zeros(3)

    if r < 0 and s < 0 and t == 0:
        raise NonDifferentiable

    # third case
    if r < 0 and s < 0:
        #print('third case')
        result = np.zeros(3)
        result[0] = dz[0]
        result[2] = dz[2] if t > 0 else 0.
        #print('result', result)
        return result

    # fourth case
    #print('fourth case')
    return fourth_case_D(r, s, t, x, y, z, dr, ds, dt)
    # mat = np.zeros((3, 3))
    # rhs = np.zeros(3)
    # # first eq.
    # mat[0, 0] = 2 * x - r
    # mat[0, 1] = 2 * y - s
    # mat[0, 2] = 2 * z - t
    # rhs[0] = x * dr + y * ds + z * dt
    # # second eq.
    # mat[1, 0] = np.exp(x / y)
    # mat[1, 1] = np.exp(x / y) * (1 - x / y)
    # mat[1, 2] = -1.
    # # third eq.
    # mat[2, 0] = y
    # mat[2, 1] = x - r
    # mat[2, 2] = 2 * z - t
    # rhs[2] = y * dr + z * dt

    # result = np.linalg.solve(mat, rhs)
    # print('result', result)
    # return result


# @njit
# def exp_pri_DT(z, x, cache):
#     """Derivative of proj. transp. on exp. primal cone."""
#     pass

exp_pri_cone = cone(exp_pri_Pi, exp_pri_D, exp_pri_D)  # exp_pri_DT)


@njit
def exp_dua_Pi(z):
    """Projection on exp. dual cone, and cache."""
    minuspi, cache = exp_pri_Pi(-z)
    return np.copy(z) + minuspi, cache


@njit
def exp_dua_D(z, x, cache):
    """Derivative of projection on exp. dual cone."""
    #res = exp_pri_D(z, x, cache)
    return np.copy(x) + exp_pri_D(-z, -x, cache)


# @njit
# def exp_dua_DT(z, x, cache):
#     """Derivative of proj. transp. on exp. dual cone."""
#     return exp_pri_DT(z, x, cache) - np.copy(x)

exp_dua_cone = cone(exp_dua_Pi, exp_dua_D, exp_dua_D)


@jit
def prod_cone_D(z, x, cones_caches):
    cur = 0
    result = np.empty_like(x)
    for (cone, n, cache) in cones_caches:
        result[cur:cur + n] = cone.D(z[cur:cur + n], x[cur:cur + n], cache)
        cur += n
    assert cur == len(z)
    return result


@jit
def prod_cone_Pi(z, cones):
    result = np.empty_like(z)
    cur = 0
    cones_caches = []
    for (cone, n) in cones:
        result[cur:cur + n], cache = cone.Pi(z[cur:cur + n])
        cones_caches.append([cone, n, cache])
        cur += n
    return result, cones_caches


prod_cone = cone(prod_cone_Pi, prod_cone_D, prod_cone_D)


# def proj_sdp(z, n):
#     z = np.copy(z)
#     if n == 0:
#         return
#     elif n == 1:
#         return np.pos(z)
#     tidx = triu_indices(n)
#     tidx = (tidx[1], tidx[0])
#     didx = diag_indices(n)

#     a = zeros((n, n))
#     a[tidx] = z
#     a = (a + transpose(a))
#     a[didx] = a[didx] / sqrt(2.)

#     w, v = linalg.eig(a)  # cols of v are eigenvectors
#     w = pos(w)
#     a = dot(v, dot(diag(w), transpose(v)))
#     a[didx] = a[didx] / sqrt(2.)
#     z = a[tidx]
#     return z


#@lru_cache(3)
@njit
def sizevec2sizemat(n):
    m = int(np.sqrt(2 * n))
    if not n == (m * (m + 1) / 2.):
        raise DimensionError
    return m


@lru_cache(3)
def tri_indices(n):

    tidx = np.triu_indices(n)
    tidx = (tidx[1], tidx[0])
    didx = np.diag_indices(n)

    return tidx, didx


@jit
def mat2vec(Z):
    """Upper tri row stacked, off diagonal multiplied by sqrt(2)."""
    n = Z.shape[0]

    tidx, didx = tri_indices(n)

    old_diag = Z[didx]
    Z *= np.sqrt(2.)
    Z[didx] = old_diag

    return Z[tidx]


@jit
def vec2mat(z):
    l = len(z)
    n = sizevec2sizemat(l)

    tidx, didx = tri_indices(n)

    Z = np.zeros((n, n))
    Z[tidx] = z
    old_diag = Z[didx]
    Z += Z.T
    Z /= np.sqrt(2.)
    Z[didx] = old_diag
    return Z


@njit
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


@jit
def semidef_cone_D(z, dz, cache):
    U, lambda_var = cache
    dZ = vec2mat(dz)
    dZ_tilde = U.T @ dZ @ U
    lambda_plus = np.maximum(lambda_var, 0.)
    lambda_minus = -np.minimum(lambda_var, 0.)
    m = len(lambda_plus)
    k = np.sum(lambda_plus > 0)
    if not (np.sum(lambda_minus > 0) + k == m):
        # TODO can't throw excep. in numba
        print('SD cone projection not differentiable!')
    dS_tilde = np.empty((m, m))

    inner(m, lambda_plus, lambda_minus, dS_tilde, dZ_tilde)

    dS = U @ dS_tilde @ U.T
    return mat2vec(dS)


@jit
def embedded_cone_D(z, dz, cache):
    """Der. of proj. on the cone of the primal-dual embedding."""
    return dz - prod_cone.D(-z, dz, cache)


@jit
def embedded_cone_Pi(z, cones, n):
    """Projection on the cone of the primal-dual embedding."""
    emb_cones = [[zero_cone, n]] + cones + [[non_neg_cone, 1]]
    v, cache = prod_cone.Pi(-z, emb_cones)
    return v + z, cache

embedded_cone = cone(embedded_cone_Pi, embedded_cone_D, embedded_cone_D)


#     def __matmul__new(self, dz):

#         dZ = vec2mat(dz)
#         dZ_tilde = self.eivec.T @ dZ @ self.eivec
#         lambda_plus = np.maximum(self.eival, 0.)
#         lambda_minus = -np.minimum(self.eival, 0.)

#         m = len(lambda_plus)
#         k = sum(lambda_plus > 0)
#         assert(sum(lambda_minus > 0) + k == m)

#         dS_tilde = np.empty((m, m))

#         dS_tilde[:k, :k] = 0.
#         dS_tilde[k:, k:] = dZ_tilde[k:, k:]
#         # assert(np.alltrue(dS_tilde[k:, k:] > 0))

#         # hard part
#         factor_nom = np.tile(lambda_plus[-k:], (m - k, 1))
#         assert(factor_nom.shape == dZ_tilde[k:, :k].shape)

#         factor_denom = np.array(factor_nom)
#         assert(np.alltrue(factor_denom > 0))

#         factor_denom_2 = np.tile(lambda_minus[:m - k], (k, 1)).T
#         assert(np.alltrue(factor_denom_2 > 0))
#         factor_denom += factor_denom_2

#         factor = factor_nom / factor_denom

#         dS_tilde[k:, :k] = dZ_tilde[k:, :k] * factor
#         dS_tilde[:k, k:] = dS_tilde[k:, :k].T

#         assert (np.allclose(dS_tilde - dS_tilde.T, 0.))

#         dS = self.eivec @ dS_tilde @ self.eivec.T
#         return mat2vec(dS)


@jit
def semidef_cone_Pi(z):
    # check_right_size(z, self.n)
    Z = vec2mat(z)
    # TODO cache this and do it only once..
    eival, eivec = np.linalg.eigh(Z)
    # print(eival, eivec)
    result = mat2vec(eivec @ np.diag(np.maximum(eival, 0.)) @ eivec.T)
    # print('S')
    # print(result)
    return result, (eivec, eival)

semi_def_cone = cone(semidef_cone_Pi, semidef_cone_D, semidef_cone_D)
