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
# import scs
# import ecos
import scipy.sparse as sp

from .cones import *
from .problem import *


def generate_dim_dict(zero_num_min=0,
                      zero_num_max=200,
                      nonneg_num_min=0,
                      nonneg_num_max=500,
                      lorentz_num_min=0,
                      lorentz_num_max=50,
                      lorentz_size_min=1,
                      lorentz_size_max=50,
                      semidef_num_min=0,
                      semidef_num_max=50,
                      semidef_size_min=1,
                      semidef_size_max=20,
                      exp_num_min=0,
                      exp_num_max=100):
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
    s = prod_cone.Pi(r, cache)
    y = s - r

    A = sp.rand(m, n, density=density, format='csc')
    A.data = np.random.uniform(min_val_entries, max_val_entries, size=A.nnz)
    # np.random.randn(
    #     A.nnz) * np.random.uniform(1., 1. + random_scale_max)
    # x = np.random.randn(n) * np.random.uniform(1., 1. + random_scale_max)
    A /= np.linalg.norm(A.data)

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
        # np.random.randn(n)
        c = np.random.uniform(min_val_entries, max_val_entries, size=n)
        x *= 0.  # same as cert of infeas
        s *= 0.
        return sp.csc_matrix(A), b, c, dim_dict, x, s, y

    else:
        raise Exception('Invalid mode.')
