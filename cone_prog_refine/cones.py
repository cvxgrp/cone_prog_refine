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

import numpy as np
import scipy.sparse as sp
from collections import namedtuple
from numba import njit, jit
from functools import lru_cache


class DimensionError(Exception):
    pass


class NonDifferentiable(Exception):
    pass


def check_non_neg_int(n):
    if not isinstance(n, int):
        raise TypeError
    if n < 0:
        raise DimensionError


def check_right_size(x, n):
    if (n > 1) and not (len(x) == n):
        raise DimensionError


cone = namedtuple('cone', ['Pi', 'D'])


@njit
def id_op(z, x, cache):
    """Identity operator on x."""
    return np.copy(x)


@njit
def free(z):
    """Projection on free cone, and cache."""
    cache = None
    return np.copy(z), cache


free_cone = cone(free, id_op)


@njit
def zero_D(z, x, cache):
    """zero operator on x."""
    return np.zeros_like(x)


@njit
def zero_Pi(z):
    """Projection on zero cone, and cache."""
    cache = None
    return np.zeros_like(z), cache

zero_cone = cone(zero_Pi, zero_D)


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


non_neg_cone = cone(non_neg_Pi, non_neg_D)


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


@jit
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

    return np.concatenate([[1.], x / norm_x]) * (norm_x + rho) / 2., cache


sec_ord_cone = cone(sec_ord_Pi, sec_ord_D)


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


prod_cone = cone(prod_cone_Pi, prod_cone_D)


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


@lru_cache(3)
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
    return dz - prod_cone.D(-z, dz, cache)


@jit
def embedded_cone_Pi(z, cones, n):
    emb_cones = [[zero_cone, n]] + cones + [[non_neg_cone, 1]]
    v, cache = prod_cone.Pi(-z, emb_cones)
    return v + z, cache

embedded_cone = cone(embedded_cone_Pi, embedded_cone_D)


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
    #check_right_size(z, self.n)
    Z = vec2mat(z)
    # TODO cache this and do it only once..
    eival, eivec = np.linalg.eigh(Z)
    # print(eival, eivec)
    result = mat2vec(eivec @ np.diag(np.maximum(eival, 0.)) @ eivec.T)
    # print('S')
    # print(result)
    return result, (eivec, eival)

semi_def_cone = cone(semidef_cone_Pi, semidef_cone_D)

# TODO
exp_pri_cone = None
exp_dua_cone = None
