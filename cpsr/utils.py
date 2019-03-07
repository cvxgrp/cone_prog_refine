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

import numpy as np
import scipy.sparse as sp

from .cones import *
from .problem import *


def generate_dim_dict(zero_num_min=10,
                      zero_num_max=50,
                      nonneg_num_min=20,
                      nonneg_num_max=100,
                      lorentz_num_min=20,
                      lorentz_num_max=100,
                      lorentz_size_min=5,
                      lorentz_size_max=20,
                      semidef_num_min=5,
                      semidef_num_max=20,
                      semidef_size_min=2,
                      semidef_size_max=10,
                      exp_num_min=2,
                      exp_num_max=10,
                      random_ecos=.25):
    result = {}
    result['z'] = int(np.random.uniform(zero_num_min,
                                        zero_num_max))
    result['l'] = int(np.random.uniform(nonneg_num_min,
                                        nonneg_num_max))
    num_q = int(np.random.uniform(lorentz_num_min,
                                  lorentz_num_max))
    result['q'] = [int(np.random.uniform(lorentz_size_min,
                                         lorentz_size_max))
                   for i in range(num_q)]
    num_s = int(np.random.uniform(semidef_num_min,
                                  semidef_num_max))
    result['s'] = [int(np.random.uniform(semidef_size_min,
                                         semidef_size_max))
                   for i in range(num_s)]
    result['ep'] = int(np.random.uniform(exp_num_min,
                                         exp_num_max))

    result['ed'] = int(np.random.uniform(exp_num_min,
                                         exp_num_max))
    if np.random.uniform() < random_ecos:
        result['s'] = []
        result['ep'] = 0
        result['ed'] = 0
    return result


def generate_problem(dim_dict=None,
                     density=None,
                     mode=None,  # TODO fix tests
                     # nondiff_point=False,
                     # random_scale_max=None,
                     min_val_entries=-1,
                     max_val_entries=1):  # TODO drop option
    """Generate random problem with given cone and density."""

    if dim_dict is None:
        dim_dict = generate_dim_dict()

    if density is None:
        density = np.random.uniform(.2, .4)

    if mode is None:
        mode = np.random.choice(['solvable', 'infeasible', 'unbounded'])

    m = (dim_dict['l'] if 'l' in dim_dict else 0) + \
        (dim_dict['z'] if 'z' in dim_dict else 0) + \
        (sum(dim_dict['q']) if 'q' in dim_dict else 0) + \
        (sum([sizemat2sizevec(el) for el in dim_dict['s']]) if 's' in dim_dict else 0) + \
        (3 * dim_dict['ep'] if 'ep' in dim_dict else 0) + \
        (3 * dim_dict['ed'] if 'ed' in dim_dict else 0)

    n = int(np.random.uniform(1, m))

    # r = np.zeros(m) if nondiff_point else
    r = np.random.uniform(min_val_entries, max_val_entries,
                          size=m)  # np.random.randn(m)

    cache = make_prod_cone_cache(dim_dict)
    s = prod_cone.Pi(r, *cache)
    y = s - r

    A = sp.rand(m, n, density=density, format='csc')
    A.data = np.random.uniform(min_val_entries, max_val_entries, size=A.nnz)
    # np.random.randn(
    #     A.nnz) * np.random.uniform(1., 1. + random_scale_max)
    # x = np.random.randn(n) * np.random.uniform(1., 1. + random_scale_max)
    A /= np.linalg.norm(A.data)
    #A *= m/n

    x = np.random.uniform(min_val_entries, max_val_entries, size=n)

    if mode == 'solvable':
        b = A@x + s
        c = -A.T@y
        return A, b, c, dim_dict,  x, s, y

    if mode == 'unbounded':
        x[x == 0] += 1
        error = A@x + s
        sparsity = np.array(A.todense() != 0, dtype=int)
        for i in range(m):
            j = np.argmax(sparsity[i, :])
            A[i, j] -= error[i] / x[j]
        assert np.allclose(A@x + s, 0.)
        # A = A - np.outer(s + A@x, x) / np.linalg.norm(x)**2  # dense...
        # c = np.random.randn(n) * np.random.uniform(1., 1. + random_scale_max)
        c = - x / (x@x)  # c / (c@x)
        assert np.allclose(c@x, -1)
        # np.random.randn(m)
        b = np.random.uniform(min_val_entries, max_val_entries, size=m)
        y *= 0.  # same as cert of unbound
        return A, b, c, dim_dict, x, s, y

    if mode == 'infeasible':
        error = A.T@y
        sparsity = np.array(A.todense() != 0, dtype=int)
        # old_nnz = A.nnz
        # D = np.array(A.todense() != 0, dtype=int)
        # B = sp.csc_matrix(np.multiply((A.T@y) / (D.T@y), D))
        # A = A - B
        for j in range(n):
            i = np.argmax(sparsity[:, j] * y**2)
            A[i, j] -= error[j] / y[i]
        assert np.allclose(A.T@y, 0.)
        # assert old_nnz == A.nnz
        # correction = A.T@y / sum(y)
        # A = A - np.outer(y, A.T@y) / np.linalg.norm(y)**2  # dense...
        # b = np.random.randn(m) * np.random.uniform(1., 1. + random_scale_max)
        b = - y / (y@y)  # - b / (b@y)
        assert np.allclose(b@y, -1)
        # np.random.randn(n)
        c = np.random.uniform(min_val_entries, max_val_entries, size=n)
        x *= 0.  # same as cert of infeas
        s *= 0.
        return sp.csc_matrix(A), b, c, dim_dict, x, s, y

    else:
        raise Exception('Invalid mode.')
