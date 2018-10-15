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
import scs
import ecos
import scipy.sparse as sp


from .cones import *
from .problem import *


def dim2cones(dim):
    """Transform dict in scs format to cones."""
    cones = []
    if 'z' in dim and dim['z'] > 0:
        cones.append([zero_cone, dim['z']])
    if 'f' in dim and dim['f'] > 0:
        cones.append([free_cone, dim['f']])
    if 'l' in dim and dim['l'] > 0:
        cones.append([non_neg_cone, dim['l']])
    if 'q' in dim:
        for q in dim['q']:
            if q > 0:
                cones.append([sec_ord_cone, q])
    if 's' in dim:
        for s in dim['s']:
            if s > 0:
                cones.append([semi_def_cone, s * (s + 1) // 2])
    if 'ep' in dim:
        for i in range(dim['ep']):
            cones.append([exp_pri_cone, 3])
    if 'ed' in dim:
        for i in range(dim['ed']):
            cones.append([exp_dua_cone, 3])
    return cones


def generate_problem(dim_dict, density=.1, mode='solvable', nondiff_point=False, random_scale_max=10.):
    """Generate random problem with given cone and density."""
    cones = dim2cones(dim_dict)
    m = sum([el[1] for el in cones])
    n = m

    r = np.zeros(m) if nondiff_point else np.random.randn(m)

    s, cache = prod_cone.Pi(r, cones)
    y = s - r
    A = sp.rand(m, n, density=density, format='csc')
    A.data = np.random.randn(
        A.nnz) * np.random.uniform(1., 1. + random_scale_max)

    x = np.random.randn(n) * np.random.uniform(1., 1. + random_scale_max)

    if mode == 'solvable':
        b = A@x + s
        c = -A.T@y
        return A, b, c, x, s, y

    if mode == 'unbounded':
        A = A - np.outer(s + A@x, x) / np.linalg.norm(x)**2  # dense...
        c = np.random.randn(n) * np.random.uniform(1., 1. + random_scale_max)
        c = -c / (c@x)
        b = np.random.randn(m)
        y *= 0.
        return sp.csc_matrix(A), b, c, x, s, y

    if mode == 'infeasible':
        A = A - np.outer(y, A.T@y) / np.linalg.norm(y)**2  # dense...
        b = np.random.randn(m) * np.random.uniform(1., 1. + random_scale_max)
        b = - b / (b@y)
        c = np.random.randn(n)
        x *= 0.
        s *= 0.
        return sp.csc_matrix(A), b, c, x, s, y

    else:
        raise Exception('Invalid mode.')
