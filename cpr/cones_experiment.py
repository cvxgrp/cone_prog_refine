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

# import ctypes
# # from ctypes import cdll, c_double, c_int, POINTER
# c_cones = ctypes.cdll.LoadLibrary(
#     '/Users/enzo/repos/cone_prog_refine/c/libcones.so')

# c_zero_p = c_cones.zero_cone_projection
# c_zero_p.argtypes = [ctypes.POINTER(ctypes.c_double),
#                      # ctypes.c_uint64,
#                      ctypes.c_int64]
# c_zero_p.restype = None


from cffi import FFI

ffi = FFI()
ffi.cdef('void zero_cone_projection(int64_t z, int64_t zero);')
ffi.cdef('void zero_cone_projection_derivative(int64_t z, int64_t x, int64_t zero);')

ffi.cdef('void non_negative_cone_projection(int64_t z, int64_t zero);')
ffi.cdef('void non_negative_cone_projection_derivative(int64_t z, int64_t x, int64_t zero);')

ffi.cdef('void second_order_cone_projection(int64_t z, int64_t zero);')
ffi.cdef('void second_order_cone_projection_derivative(int64_t z, int64_t dz, int64_t pi_z, int64_t zero);')

ffi.cdef('void exp_cone_projection(int64_t z);')
ffi.cdef('int exp_cone_projection_derivative(int64_t z, int64_t dz, int64_t pi_z);')

ffi.cdef('int compute_jacobian_exp_cone(int64_t result, double mu_star, double x_star, double y_star, double z_star);')


import os
dirname = os.path.dirname(__file__)
libpath = os.path.join(dirname, '../c/libcones.so')
C = ffi.dlopen(libpath)

c_zero_p = C.zero_cone_projection
c_zero_p_d = C.zero_cone_projection_derivative

c_non_neg_p = C.non_negative_cone_projection
c_non_neg_p_d = C.non_negative_cone_projection_derivative

c_sec_ord_p = C.second_order_cone_projection
c_sec_ord_p_d = C.second_order_cone_projection_derivative

c_exp_p = C.exp_cone_projection
c_exp_p_d = C.exp_cone_projection_derivative

c_compute_jacobian_exp_cone = C.compute_jacobian_exp_cone


# TEST
z = np.arange(10)
print(z)
c_zero_p(z.ctypes.data, len(z))
print(z)
assert np.all(z == 0.)
#######


# def check_non_neg_int(n):
#     if not isinstance(n, int):
#         raise TypeError
#     if n < 0:
#         raise DimensionError


# def check_right_size(x, n):
#     if (n > 1) and not (len(x) == n):
#         raise DimensionError


cone = namedtuple('cone', ['Pi', 'D', 'DT', 'isin'])


@nb.jit(nb.float64[:](nb.float64[:], nb.float64[:]))
def id_op(z, x):
    """Identity operator on x."""
    return x  # np.copy(x)


@nb.jit(nb.float64[:](nb.float64[:]))
def free(z):
    """Projection on free cone, and cache."""
    return z  # np.copy(z)


free_cone = cone(free, id_op, id_op,
                 lambda z: True)
free_cone_cached = cone(lambda z, cache: free_cone.Pi(z),
                        lambda z, x, cache: free_cone.D(z, x),
                        lambda z, x, cache: free_cone.D(z, x),
                        lambda z: True)


@nb.jit(nb.float64[:](nb.float64[:], nb.float64[:]))
def zero_D(z, x):
    """zero operator on x."""
    # x[:] = 0.
    c_zero_p_d(z.ctypes.data, x.ctypes.data, len(z))

    return x  # np.zeros_like(x)


@nb.jit(nb.float64[:](nb.float64[:]))
def zero_Pi(z):
    """Projection on zero cone, and cache."""
    # z[:] = 0.

    # z.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    c_zero_p(z.ctypes.data, len(z))

    # c_zero_p(
    #     # z.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    #     # z.ctypes.data,
    #     CPointer(float64(z.ctypes.data)),
    #     #nb.carray(z.ctypes.data, (len(z),)),
    #     len(z)
    # )

    return z  # np.zeros_like(z)


zero_cone = cone(zero_Pi, zero_D, zero_D,
                 lambda z: np.all(z == 0.))
zero_cone_cached = cone(lambda z, cache: zero_cone.Pi(z),
                        lambda z, x, cache: zero_cone.D(z, x),
                        lambda z, x, cache: zero_cone.D(z, x),
                        lambda z: np.all(z == 0.))


@nb.jit(nb.float64[:](nb.float64[:], nb.float64[:]), nopython=True)
def non_neg_D(z, x):
    # assert len(x) == len(z)
    # result = np.copy(x)
    # x[z <= 0] = 0.

    c_non_neg_p_d(z.ctypes.data, x.ctypes.data, len(z))

    return x


@nb.jit(nb.float64[:](nb.float64[:]), nopython=True)
def non_neg_Pi(z):
    """Projection on non-negative cone, and cache."""
    # cache = np.zeros(1)
    # z[z <= 0.] = 0.

    c_non_neg_p(z.ctypes.data, len(z))

    return z  # np.maximum(z, 0.)


non_neg_cone = cone(non_neg_Pi, non_neg_D, non_neg_D,
                    lambda z: np.min(z) >= 0.)
non_neg_cone_cached = cone(lambda z, cache: non_neg_cone.Pi(z),
                           lambda z, x, cache: non_neg_cone.D(z, x),
                           lambda z, x, cache: non_neg_cone.D(z, x),
                           lambda z: np.min(z) >= 0.)


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
def sec_ord_D(z, dz, pi_z):
    """Derivative of projection on second order cone."""

    # logic for simple cases

    # t, x = z[0], z[1:]

    # norm = np.linalg.norm(z[1:])

    # if norm <= z[0]:
    #     return dz

    # if norm <= -z[0]:
    #     dz[:] = 0.
    #     return dz

    # dot_val = dz[1:] @ x
    # old_dzzero = dz[0]

    # dz[0] = (dz[0] * norm + dot_val) / (2 * norm)
    # dz[1:] = dz[1:] * (t + norm) / (2 * norm) + \
    #     x * ((old_dzzero - t * dot_val / norm**2) / (2 * norm))

    c_sec_ord_p_d(z.ctypes.data, dz.ctypes.data, pi_z.ctypes.data, len(z))

    return dz


@nb.jit(nb.float64[:](nb.float64[:], nb.float64[:]))
def sec_ord_Pi(z, cache):
    """Projection on second-order cone."""

    # x, rho = z[1:], z[0]
    # # cache this?
    # norm_x = np.linalg.norm(x)
    # # cache = np.zeros(1)
    # # cache[0] = norm_x

    # if norm_x <= rho:
    #     return np.copy(z)

    # if norm_x <= -rho:
    #     return np.zeros_like(z)

    # result = np.empty_like(z)

    # result[0] = 1.
    # result[1:] = x / norm_x
    # result *= (norm_x + rho) / 2.

    c_sec_ord_p(z.ctypes.data, len(z))

    cache[:] = z

    return z


def secord_isin(z):
    err = z[0] - np.linalg.norm(z[1:])
    print('err', err)
    return np.isclose(min(err, 0.), 0.)


sec_ord_cone = cone(sec_ord_Pi, sec_ord_D, sec_ord_D,
                    secord_isin)


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


# @nb.jit(nb.float64(nb.float64, nb.float64, nb.float64, nb.float64))
# def newton_exp_onz(rho, y_hat, z_hat, w):
#     t = max(max(w - z_hat, -z_hat), 1e-6)
#     for iter in np.arange(0, 100):
#         f = (1 / rho**2) * t * (t + z_hat) - y_hat / rho + np.log(t / rho) + 1
#         fp = (1 / rho**2) * (2 * t + z_hat) + 1 / t

#         t = t - f / fp
#         if t <= -z_hat:
#             t = -z_hat
#             break
#         elif t <= 0:
#             t = 0
#             break
#         elif np.abs(f) < 1e-6:
#             break
#     return t + z_hat


# @nb.jit(nb.float64[:](nb.float64[:], nb.float64, nb.float64))
# def solve_with_rho(v, rho, w):
#     x = np.zeros(3)
#     x[2] = newton_exp_onz(rho, v[1], v[2], w)
#     x[1] = (1 / rho) * (x[2] - v[2]) * x[2]
#     x[0] = v[0] - rho
#     return x


# @nb.jit(nb.types.Tuple((nb.float64, nb.float64[:]))(nb.float64[:],
#                                                     nb.float64, nb.float64[:]),
#         nopython=True)
# def calc_grad(v, rho, warm_start):
#     x = solve_with_rho(v, rho, warm_start[1])
#     if x[1] == 0:
#         g = x[0]
#     else:
#         g = (x[0] + x[1] * np.log(x[1] / x[2]))
#     return g, x


# @nb.jit(nb.types.UniTuple(nb.float64, 2)(nb.float64[:]))
# def get_rho_ub(v):
#     lb = 0
#     rho = 2**(-3)
#     g, z = calc_grad(v, rho, v)
#     while g > 0:
#         lb = rho
#         rho = rho * 2
#         g, z = calc_grad(v, rho, z)
#     ub = rho
#     return ub, lb


# @nb.jit(nb.types.UniTuple(nb.float64[:], 2)(nb.float64[:]))
# def fourth_case_brendan(z):
#     x = np.copy(z)
#     ub, lb = get_rho_ub(x)
#     # print('initial ub, lb', ub, lb)
#     for i in range(0, 100):
#         rho = (ub + lb) / 2
#         g, x = calc_grad(z, rho, x)
#         # print('g, x', g, x)
#         if g > 0:
#             lb = rho
#         else:
#             ub = rho
#         # print('new ub, lb', ub, lb)
#         if ub - lb < 1e-8:
#             break
#     # print('result:', x)
#     return x, np.copy(x)


# # Here it is my work on exp proj D
# @nb.jit(nb.float64[:, :](nb.float64, nb.float64, nb.float64, nb.float64,
#                          nb.float64, nb.float64))
# def make_mat(r, s, t, x, y, z):
#     mat = np.zeros((3, 3))
#     # first eq.
#     mat[0, 0] = 2 * x - r
#     mat[0, 1] = 2 * y - s
#     mat[0, 2] = 2 * z - t
#     mat[1, 0] = np.exp(x / y)
#     mat[1, 1] = np.exp(x / y) * (1 - x / y)
#     mat[1, 2] = -1.
#     mat[2, 0] = y
#     mat[2, 1] = x - r
#     mat[2, 2] = 2 * z - t
#     return mat

# Here it is my work on exp proj D


# @nb.jit(nb.float64[:, :](nb.float64, nb.float64, nb.float64, nb.float64,
#                          nb.float64, nb.float64))
# def make_mat_two(r, s, t, x, y, z):
#     mat = np.zeros((3, 3))
#     # first eq.
#     u = -(r - x)
#     v = -(s - y)
#     w = -(t - z)

#     mat[0, 0] = 2 * x - r
#     mat[0, 1] = 2 * y - s
#     mat[0, 2] = 2 * z - t

#     # mat[1, 0] = np.exp(x / y)
#     # mat[1, 1] = np.exp(x / y) * (1 - x / y)
#     # mat[1, 2] = -1.

#     mat[1, 0] = np.exp(v / u) * (1 - v / u)
#     mat[1, 1] = np.exp(v / u)
#     mat[1, 2] = np.e

#     mat[2, 0] = y
#     mat[2, 1] = x - r
#     mat[2, 2] = 2 * z - t
#     return mat


# @nb.jit(nb.float64[:](nb.float64, nb.float64, nb.float64, nb.float64,
#                       nb.float64, nb.float64))
# def make_error(r, s, t, x, y, z):
#     error = np.zeros(3)
#     error[0] = x * (x - r) + y * (y - s) + z * (z - t)
#     error[1] = y * np.exp(x / y) - z
#     error[2] = y * (x - r) + z * (z - t)
#     print('error', error)
#     return error


# @nb.jit(nb.float64[:](nb.float64, nb.float64, nb.float64, nb.float64,
#                       nb.float64, nb.float64))
# def make_error_two(r, s, t, x, y, z):
#     error = np.zeros(3)
#     error[0] = x * (x - r) + y * (y - s) + z * (z - t)
#     u = -(r - x)
#     v = -(s - y)
#     w = -(t - z)
#     error[1] = np.e * w + u * np.exp(v / u)
#     error[2] = y * (x - r) + z * (z - t)
#     # print('error', error)
#     return error


# @nb.jit(nb.float64[:](nb.float64, nb.float64, nb.float64, nb.float64,
#                       nb.float64, nb.float64))
# def make_rhs(x, y, z, dr, ds, dt):
#     rhs = np.zeros(3)
#     rhs[0] = x * dr + y * ds + z * dt
#     rhs[2] = y * dr + z * dt
#     return rhs


# @nb.jit(nb.float64[:](nb.float64, nb.float64, nb.float64, nb.float64,
#                       nb.float64, nb.float64, nb.float64, nb.float64, nb.float64))
# def fourth_case_D(r, s, t, x, y, z, dr, ds, dt):

#     test = np.zeros(3)
#     test[0], test[1], test[2] = r, s, t

#     u = x - r

#     if y < 1E-12:
#         return np.zeros(3)

#     # if y > -u and not y == 0.:  # temporary fix?
#         # print('computing error with e^(x/y)')
#     error = make_error(r, s, t, x, y, z)
#     # else:
#     # print('computing error with e^(v/u)')
#     #    error = make_error_two(r, s, t, x, y, z)

#     # error = make_error(r, s, t, x, y, z)
#     rhs = make_rhs(x, y, z, dr, ds, dt)
#     # print('base rhs', rhs)
#     if np.any(np.isnan(error)) or np.any(np.isinf(error)) or np.any(np.isnan(rhs)):
#         return np.zeros(3)

#     # if y > -u and not y == 0.:  # temporary fix?
#         # print('solving system with e^(x/y)')
#     result = np.linalg.solve(make_mat(r, s, t, x, y, z) + np.eye(3) * 1E-8,
#                              rhs - error)
#     # else:
#     # print('solving system with e^(v/u)')
#     #    result = np.linalg.solve(make_mat_two(r, s, t, x, y, z),  # + np.eye(3) * 1E-8,
#     #                             rhs - error)

#     if np.any(np.isnan(result)):
#         # raise Exception('Exp cone derivative error.')
#         return np.zeros(3)
#     # print('result', result)
#     return result

# MAXITER = 100
# STEP = 0.5


# @nb.jit(nb.float64[:](nb.float64[:]))
# def fourth_case_enzo(z_var):
#     # VERBOSE = False

#     # print('fourth case Enzo')
#     # if VERBOSE:
#     #    print('projecting (%f, %f, %f)' % (z_var[0], z_var[1], z_var[2]))
#     real_result, _ = fourth_case_brendan(z_var)
#     # print('brendan result: (x,y,z)', real_result)
#     z_var = np.copy(z_var)
#     r = z_var[0]
#     s = z_var[1]
#     t = z_var[2]
#     # print('brendan result: (u,v,w)', real_result - z_var)

#     x, y, z = real_result

#     # x, y, z = 1, 1, np.e
#     # if y < 1E-12:
#     #     y = 1E-12

#     for i in range(10):

#         # if y < 1e-14:
#         #     return np.zeros(3)
#         u = x - r
#         # assert y >= 0
#         # assert u <= 0
#         if (y <= 0. and u >= 0.):
#             return np.zeros(3)

#         if y > -u and not y == 0.:
#             # print('computing error with e^(x/y)')
#             error = make_error(r, s, t, x, y, z)
#         else:
#             # print('computing error with e^(v/u)')
#             if u == 0:
#                 break
#             error = make_error_two(r, s, t, x, y, z)

#         # print('error:', error)
#         # print('error norm: %.2e' % np.linalg.norm(error))

#         # if VERBOSE:
#         # print('iter %d, max |error| = %g' % (i, np.max(np.abs(error))))

#         print(np.max(np.abs(error)))

#         if np.max(np.abs(error)) < 1e-15:
#             # print(np.max(np.abs(error)))
#             # print('converged!')
#             break

#         if np.any(np.isnan(error)):
#             raise Exception("Exponential cone error")

#         if y > -u and not y == 0.:
#             # print('computing correction with e^(x/y)')
#             correction = np.linalg.solve(
#                 make_mat(r, s, t, x, y, z) + np.eye(3) * 1E-8,
#                 -error)
#         else:
#             # print('computing correction with e^(v/u)')
#             correction = np.linalg.solve(
#                 make_mat_two(r, s, t, x, y, z) + np.eye(3) * 1E-8,
#                 -error)

#         # print('correction', correction)

#         if np.any(np.isnan(correction)):
#             raise Exception("Exponential cone error")

#         x += correction[0]
#         y += correction[1]
#         z += correction[2]

#     result = np.empty(3)
#     result[0] = x
#     result[1] = y
#     result[2] = z

#     # if VERBOSE:
#     #    print('result = (%f, %f, %f)' % (result[0], result[1], result[2]))

#     return result


# @nb.jit(nb.float64[:](nb.float64[:]))
# def fourth_case_enzo_two(z_var):
#     # # VERBOSE = False

#     # # print('fourth case Enzo')
#     # # if VERBOSE:
#     # #    print('projecting (%f, %f, %f)' % (z_var[0], z_var[1], z_var[2]))
#     # real_result, _ = fourth_case_brendan(z_var)
#     # # print('brendan result: (x,y,z)', real_result)
#     # z_var = np.copy(z_var)
#     r = z_var[0]
#     s = z_var[1]
#     t = z_var[2]
#     # # print('brendan result: (u,v,w)', real_result - z_var)

#     # x, y, z = real_result

#     #x, y, z = 1, 1, np.e
#     # x, y, z = r, s, s * np.exp(r / s)
#     x, y, z = r, s, t
#     # if y < 1E-12:
#     #     y = 1E-12

#     for i in range(100):

#         # if y < 1e-14:
#         #     return np.zeros(3)
#         u = x - r
#         # assert y >= 0
#         # assert u <= 0
#         if (y <= 0. and u >= 0.):
#             return np.zeros(3)

#         error = make_error(r, s, t, x, y, z)

#         # print('error:', error)
#         # print('error norm: %.2e' % np.linalg.norm(error))

#         # if VERBOSE:
#         # print('iter %d, max |error| = %g' % (i, np.max(np.abs(error))))

#         print(np.max(np.abs(error)))

#         if np.max(np.abs(error)) < 1e-15:
#             # print(np.max(np.abs(error)))
#             # print('converged!')
#             break

#         if np.any(np.isnan(error)):
#             raise Exception("Exponential cone error")

#         correction = np.linalg.solve(
#             make_mat(r, s, t, x, y, z),  # + np.eye(3) * 1E-5,
#             -error)

#         if np.any(np.isnan(correction)):
#             raise Exception("Exponential cone error")

#         x += correction[0]  # * 0.8
#         y += correction[1]  # * 0.8
#         z += correction[2]  # * 0.8

#     result = np.empty(3)
#     result[0] = x
#     result[1] = y
#     result[2] = z

#     # if VERBOSE:
#     #    print('result = (%f, %f, %f)' % (result[0], result[1], result[2]))

#     return result

# @nb.njit()
# def compute_error(z, pi_z, rho):

#     r, s, t = pi_z

#     error = np.zeros(4)
#     alpha = r / s
#     beta = np.exp(alpha)
#     gamma = rho * beta / s

#     error[:3] = pi_z - z
#     error[0] += rho * beta
#     error[1] += rho * beta * (1 - alpha)
#     error[2] -= rho
#     error[3] = s * beta - t

#     return error


@nb.njit()
def conjgrad(A, b, x, maxiter):
    r = b - A @ x
    p = r
    rsold = r @ r

    for i in range(maxiter):
        Ap = A @ p
        alpha = rsold / (p.T @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = r @r
        # if np.sqrt(rsnew) < 1e-10:
        #     break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    return x


@nb.njit()
def g(z):
    r, s, t = z
    return s * np.exp(r / s) - t


@nb.njit()
def grad_g(z):
    r, s, t = z
    result = np.empty(3)
    result[0] = np.exp(r / s)
    result[1] = np.exp(r / s) * (1 - r / s)
    result[2] = -1
    return result


@nb.njit()
def hessian_g(z):
    r, s, t = z
    result = np.zeros((3, 3))
    result[0, 0] = 1 / s
    result[1, 1] = r**2 / s**3
    result[0, 1] = -r / s**2
    result[1, 0] = result[0, 1]
    result *= np.exp(r / s)
    return result


@nb.njit()
def h(z, z_0):
    u, v, w = z - z_0
    return -u * np.exp(v / u) - np.e * w


@nb.njit()
def grad_h(z, z_0):
    u, v, w = z - z_0
    result = np.empty(3)
    result[0] = np.exp(v / u) * (v / u - 1)
    result[1] = -np.exp(v / u)
    result[2] = -np.e
    return result


@nb.njit()
def hessian_h(z):
    u, v, w = z - z_0
    result = np.zeros((3, 3))
    result[0, 0] = v**2 / u**3
    result[1, 1] = 1 / u
    result[0, 1] = -v / u**2
    result[1, 0] = result[0, 1]
    result *= -np.exp(v / u)
    return result


@nb.njit()
def cone_newton_projection(z, pi_z, rho, mu):
    """Calculations from Parikh Boyd '14."""

    matrix = np.zeros((4, 4))
    error = np.zeros(4)
    step = 1.

    for i in range(20):

        r, s, t = pi_z
        u, v, w = pi_z - z
        print('z iterate', pi_z)
        print('rho iterate', rho)
        print('mu iterate', mu)

        error[:3] = pi_z - z + rho * grad_g(z)

        # alpha = r / s
        # beta = np.exp(alpha)
        # gamma = rho * beta / s

        # alpha_two = v / u
        # beta_two = np.exp(alpha_two)
        # gamma_two = mu * beta_two / u

        # error[:3] = pi_z - z
        # error[0] += rho * beta
        # error[1] += rho * beta * (1 - alpha)
        # error[2] -= rho

        # error[0] += mu * beta_two * (alpha_two - 1)
        # error[1] -= mu * beta_two
        # error[2] -= mu * np.e

        # error[3] = s * beta - t
        # error[4] = -u * beta_two - np.e * w

        print('error', error)

        #print('extra error', np.e * w + u * np.exp(v / u))

        if np.max(np.abs(error)) < 1E-15:
            break

        # if (y_star == 0){
        #     / *Can't compute derivative.* /
        #     return 0
        # }

        matrix[:, :] = np.zeros((4, 4))

        matrix[:3, :3] += np.eye(3)
        matrix[:3, :3] += rho * hessian_g(pi_z)
        matrix[:3, 3] += grad_g(pi_z)
        matrix[3, :3] = matrix[:3, 3]

        # matrix[0, 0] = 1 + gamma
        # matrix[0, 1] = - gamma * alpha
        # matrix[0, 2] = 0
        # matrix[0, 3] = beta

        # matrix[1, 0] = matrix[0, 1]
        # matrix[1, 1] = 1 + gamma * alpha * alpha
        # matrix[1, 2] = 0
        # matrix[1, 3] = (1 - alpha) * beta

        # matrix[2, 0] = 0
        # matrix[2, 1] = 0
        # matrix[2, 2] = 1
        # matrix[2, 3] = -1

        # matrix[3, 0] = beta
        # matrix[3, 1] = matrix[1, 3]
        # matrix[3, 2] = -1
        # matrix[3, 3] = 0

        # matrix[0, 0] += -gamma_two * alpha_two * alpha_two
        # matrix[0, 1] += -gamma_two * alpha_two
        # matrix[1, 0] = matrix[0, 1]
        # matrix[1, 1] += -gamma_two

        # matrix[0, 4] = - beta_two
        # matrix[1, 4] = beta_two * (alpha_two - 1)
        # matrix[2, 4] = -np.e
        # matrix[3, 4] = 0

        # matrix[4, :] = matrix[:, 4]

        correction = np.linalg.solve(matrix,  # + np.eye(4) * 1E-8,
                                     -error)  # ,
        # np.zeros(4),
        # maxiter=2)

        #print('correction', correction)

        delta_s = correction[1]
        delta_r = correction[0]

        max_step = -(s / delta_s) * 0.9 if delta_s < 0 else step

        # max_step = min(
        #    max_step, -((r - z[0]) / delta_r)) * 0.9 if delta_r > 0 else max_step
        #print('max step', max_step)

        assert(max_step > 0)

        # old_errsize = np.max(np.abs(error))

        mystep = min(step, max_step)

        pi_z += mystep * correction[:3]
        rho += mystep * correction[3]
        #mu += mystep * correction[4]

        # new_errsize = np.max(np.abs(compute_error(z, pi_z, rho)))
        #print('error_size', new_errsize)

        # i = 0
        # while new_errsize > old_errsize:
        #     mystep /= 2.
        #     pi_z -= mystep * correction[:3]
        #     rho -= mystep * correction[3]

        #     new_errsize = np.max(np.abs(compute_error(z, pi_z, rho)))
        #     #print('error_size', new_errsize)
        #     i += 1
        #     if i > 5:
        #         break

        # print('error_size',
        #       np.max(np.abs(compute_error(z, pi_z, rho))))

        assert(pi_z[1] > 0)
        #assert(pi_z[0] - z[0] < 0)

    return pi_z


@nb.jit(nb.float64[:](nb.float64[:], nb.float64[:]), nopython=True)
def exp_pri_Pi(z, cache):
    """Projection on exp. primal cone, and cache."""
    z = np.copy(z)
    r = z[0]
    s = z[1]
    t = z[2]

    # temporary...
    # if np.linalg.norm(z) < 1E-14:
    #     cache[:3] = 0.
    #     return z

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

    # pi = z  # fourth_case_enzo_two(z)
    init_z = np.zeros(3)
    init_z[0] = z[0] - 1
    init_z[1] = 1
    init_z[2] = np.exp(1)
    pi = cone_newton_projection(z, pi_z=init_z, rho=0, mu=0)

    cache[:3] = pi
    return pi


# @nb.jit(nb.float64[:](nb.float64[:], nb.float64[:]), nopython=True)
# def exp_pri_Pi(z, cache):

#     c_exp_p(z.ctypes.data)
#     cache[:] = z
#     return z

# CONE_THRESH = 1E-10


# @nb.njit()
# def isin_kexp(r, s, t):
#     return ((s * np.exp(r / s) - t <= CONE_THRESH and s > 0) or
#             (r <= 0 and np.abs(s) <= CONE_THRESH and t >= 0))


# @nb.njit()
# def isin_minus_kexp_star(r, s, t):
#     return ((-r < 0 and r * np.exp(s / r) + np.e * t <= CONE_THRESH) or
#             (np.abs(r) <= CONE_THRESH and -s >= 0 and -t >= 0))


# @nb.njit()
# def compute_jacobian_exp_cone(matrix, mu_star,
#                               x_star, y_star, z_star):
#     """From BMB'18 appendix C."""

#     if y_star == 0.:
#         # print('z', x, y, z)
#         print('pi z', x_star, y_star, z_star)
#         raise Exception("y_star = 0.")
#         # return np.zeros(3)

#     alpha = x_star / y_star
#     beta = np.exp(alpha)
#     gamma = mu_star * beta / y_star

#     matrix[0, 0] = 1 + gamma
#     matrix[0, 1] = - gamma * alpha
#     matrix[0, 2] = 0
#     matrix[0, 3] = beta

#     matrix[1, 0] = matrix[0, 1]
#     matrix[1, 1] = 1 + gamma * alpha * alpha
#     matrix[1, 2] = 0
#     matrix[1, 3] = (1 - alpha) * beta

#     matrix[2, 0] = 0
#     matrix[2, 1] = 0
#     matrix[2, 2] = 1
#     matrix[2, 3] = -1

#     matrix[3, 0] = beta
#     matrix[3, 1] = matrix[1, 3]
#     matrix[3, 2] = -1
#     matrix[3, 3] = 0

#     return np.linalg.inv(matrix)


# @nb.njit()
# def fourth_case_D_new(z, z_star, dz):
#     """From BMB'18 appendix C."""

#     jacobian = np.zeros((4, 4))

#     success = c_compute_jacobian_exp_cone(jacobian.ctypes.data, z_star[2] - z[2],
#                                           z_star[0], z_star[1], z_star[2])

#     if not success:
#         raise Exception('Exp cone derivative error')
#     # jacobian = compute_jacobian_exp_cone(
#     #     matrix, z_star[2] - z[2],
#     #     z_star[0], z_star[1], z_star[2])

#     return jacobian[:3, :3] @ dz


@nb.jit(nb.float64[:](nb.float64[:], nb.float64[:], nb.float64[:]), nopython=True)
def exp_pri_D(z_0, dz, cache):
    """Derivative of proj. on exp. primal cone."""

    # if isin_kexp(r, s, t):
    #     return np.copy(dz)

    # if isin_minus_kexp_star(r, s, t):
    #     return np.zeros(3)

    # # first case
    # # if on the cone boundary, non-diff
    # # if (s > 0 and s * np.exp(r / s) == t) or \
    # #         (r <= 0 and s == 0 and t >= 0):  # or \
    # #         # (r <= 0 and s == 0 and t == 0):
    # #     # raise NonDifferentiable
    # #     print('case 1')
    # #     # return np.zeros(3)

    # # # if (s > 0 and s * np.exp(r / s) < t):
    # # #     # print('first case')
    # # #     return np.copy(dz)

    # # # second case
    # # # if on cone bound, then non-diff
    # # if (-r < 0 and r * np.exp(s / r) == -np.exp(1) * t) or \
    # #         (r == 0 and -s >= 0 and -t >= 0):  # or \
    # #     # (r == 0 and -s >= 0 and -t == 0):
    # #     # raise NonDifferentiable
    # #     print('case 2')
    # #     # return np.zeros(3)

    # # if (-r < 0 and r * np.exp(s / r) < -np.exp(1) * t):  # or \
    # #        # (r == 0 and -s > 0 and -t > 0):
    # #     # print('second case')
    # #     print('case 3')
    # #     # return np.zeros(3)

    # # if r < 0 and s < 0 and t == 0:
    # #     # raise NonDifferentiable
    # #     print('case 4')
    # #     # return np.zeros(3)

    # # third case
    # if r < 0 and s < 0:
    #     # print('third case')
    #     result = np.zeros(3)
    #     result[0] = dz[0]
    #     result[2] = dz[2] if t > 0 else 0.
    #     # print('result', result)
    #     return result

    if c_exp_p_d(z_0.ctypes.data, dz.ctypes.data, cache.ctypes.data):
        return np.copy(dz)
    else:
        # return np.zeros(3)
        raise Exception('Exp cone derivative error')

    # r = z_0[0]
    # s = z_0[1]
    # t = z_0[2]

    # dr = dz[0]
    # ds = dz[1]
    # dt = dz[2]

    # # projection of z_0
    # x = cache[0]
    # y = cache[1]
    # z = cache[2]

    # fourth case
    # fourth = fourth_case_D(r, s, t, x, y, z, dr, ds, dt)

    # jacobian = np.zeros((4, 4))
    # # z, z_star, dz = z_0, cache, dz
    # success = c_compute_jacobian_exp_cone(jacobian.ctypes.data,
    #                                       cache[2] - z_0[2],
    #                                       cache[0], cache[1], cache[2])

    # if not success:
    #     raise Exception('Exp cone derivative error')
    # # jacobian = compute_jacobian_exp_cone(
    # #     matrix, z_star[2] - z[2],
    # #     z_star[0], z_star[1], z_star[2])

    # return jacobian[:3, :3] @ dz

    # fourth = fourth_case_D_new(z_0, cache, dz)
    # # assert not True in np.isnan(fourth)
    # return fourth


def isin_exppri(z):
    if z[1] > 0:
        print('margin (< 0)', (z[1] * np.exp(z[0] / z[1]) - z[2]))
    return ((z[1] > 0) and (np.isclose(max((z[1] * np.exp(z[0] / z[1]) - z[2]),
                                           0.), 0.)) or
            ((z[0] <= 0) and (z[1] == 0) and (z[2] >= 0)))


exp_pri_cone = cone(exp_pri_Pi, exp_pri_D, exp_pri_D, isin_exppri)


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


def isin_expdua(z):
    if z[0] < 0:
        print('margin (< 0)', -z[0] * np.exp(z[1] / z[0]) - np.e * z[2])
    return ((z[0] < 0) and np.isclose(max(-z[0] * np.exp(z[1] / z[0])
                                          - np.e * z[2], 0.), 0.) or
            ((z[0] == 0.) and (z[1] >= 0) and (z[2] >= 0)))

exp_dua_cone = cone(exp_dua_Pi, exp_dua_D, exp_dua_D,
                    isin_expdua)


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


@nb.njit()
def Jacobi(A):
    n = A.shape[0]            # matrix size #columns = #lines
    maxit = 100                   # maximum number of iterations
    eps = 1.0e-14             # accuracy goal
    pi = np.pi
    info = 0                     # return flag
    ev = np.zeros(n)  # ,float)     # initialize eigenvalues
    U = np.zeros((n, n))  # ,float) # initialize eigenvector
    for i in range(0, n):
        U[i, i] = 1.0

    numit = int(np.log(n))

    for t in range(0, maxit):
        s = 0    # compute sum of off-diagonal elements in A(i,j)
        for i in range(0, n):
            s = s + np.sum(np.abs(A[i, (i + 1):n]))
        # print(s)
        if (s < eps):  # diagonal form reached
            info = t
            for i in range(0, n):
                ev[i] = A[i, i]
            break
        else:
            # average value of off-diagonal elements
            limit = s / (n * (n - 1) / 2.0)
            for i in range(0, n - 1):       # loop over lines of matrix
                for j in range(i + 1, n):  # loop over columns of matrix
                    # determine (ij) such that |A(i,j)| larger than average
                    if (np.abs(A[i, j]) >= limit):
                                                      # value of off-diagonal
                                                      # elements
                        # denominator of Eq. (3.61)
                        denom = A[i, i] - A[j, j]
                        if (np.abs(denom) < eps):
                            phi = pi / 4         # Eq. (3.62)
                        else:
                            # Eq. (3.61)
                            phi = 0.5 * np.arctan(2.0 * A[i, j] / denom)
                        si = np.sin(phi)
                        co = np.cos(phi)
                        for k in range(i + 1, j):
                            store = A[i, k]
                            A[i, k] = A[i, k] * co + A[k, j] * si  # Eq. (3.56)
                            A[k, j] = A[k, j] * co - store * si  # Eq. (3.57)
                        for k in range(j + 1, n):
                            store = A[i, k]
                            A[i, k] = A[i, k] * co + A[j, k] * si  # Eq. (3.56)
                            A[j, k] = A[j, k] * co - store * si  # Eq. (3.57)
                        for k in range(0, i):
                            store = A[k, i]
                            A[k, i] = A[k, i] * co + A[k, j] * si
                            A[k, j] = A[k, j] * co - store * si
                        store = A[i, i]
                        A[i, i] = A[i, i] * co * co + 2.0 * A[i, j] * \
                            co * si + A[j, j] * si * si  # Eq. (3.58)
                        A[j, j] = A[j, j] * co * co - 2.0 * A[i, j] * \
                            co * si + store * si * si  # Eq. (3.59)
                        # Eq. (3.60)
                        A[i, j] = 0.0
                        for k in range(0, n):
                            store = U[k, j]
                            U[k, j] = U[k, j] * co - U[k, i] * si  # Eq. (3.66)
                            U[k, i] = U[k, i] * co + store * si  # Eq. (3.67)
        info = -t  # in case no convergence is reached set info to a negative value "-t"
    return ev, U, t


@nb.jit(nb.float64[:](nb.float64[:], nb.float64[:],
                      nb.float64[:]), nopython=True)
def semidef_cone_Pi(z, cache_eivec, cache_eival):

    Z = vec2mat(z)
    eival, eivec = np.linalg.eigh(Z)
    #eival, eivec, _ = Jacobi(Z)
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


semi_def_cone = cone(semidef_cone_Pi, semidef_cone_D, semidef_cone_D,
                     None)


@nb.jit(nb.float64[:](nb.float64[:], nb.float64[:],
                      nb.types.Tuple((nb.float64[:], nb.float64[:]))))
def semidef_cone_D_single_cache(z, dz, cache):
    return semidef_cone_D(z, dz, cache[0], cache[1])


@nb.jit(nb.float64[:](nb.float64[:],
                      nb.types.Tuple((nb.float64[:], nb.float64[:]))))
def semidef_cone_Pi_single_cache(z, cache):
    return semidef_cone_Pi(z, cache[0], cache[1])


def semidef_isin(z):
    min_eival = np.min(np.linalg.eigh(vec2mat(z))[0])
    print('min_eival', min_eival)
    return np.isclose(min(min_eival, 0.), 0.)

# used as test harness for semi-def functions
semi_def_cone_single_cache = cone(semidef_cone_Pi_single_cache, semidef_cone_D_single_cache,
                                  semidef_cone_D_single_cache,
                                  semidef_isin)

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


prod_cone = cone(prod_cone_Pi, prod_cone_D, prod_cone_D, None)


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


embedded_cone = cone(embedded_cone_Pi, embedded_cone_D, embedded_cone_D, None)
