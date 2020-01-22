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
"""

#__all__ = ['solve']

import numpy as np
import scs
import ecos
import time

from .problem import *
from .refine import *


class SolverError(Exception):
    pass


def scs_solve(A, b, c, dim_dict, init_z=None, **kwargs):
    """Wraps scs.solve for convenience."""
    scs_cones = {'l': dim_dict['l'] if 'l' in dim_dict else 0,
                 'q': dim_dict['q'] if 'q' in dim_dict else [],
                 's': dim_dict['s'] if 's' in dim_dict else [],
                 'ep': dim_dict['ep'] if 'ep' in dim_dict else 0,
                 'ed': dim_dict['ed'] if 'ed' in dim_dict else 0,
                 'f': dim_dict['z'] if 'z' in dim_dict else 0}
    #print('scs_cones', scs_cones)
    sol = scs.solve({'A': A, 'b': b,
                     'c': c},
                    cone=scs_cones,
                    **kwargs)
    info = sol['info']

    if info['statusVal'] > 0:
        z = xsy2z(sol['x'], sol['s'], sol['y'], tau=1., kappa=0.)

    if info['statusVal'] < 0:
        x = np.zeros_like(sol['x']) \
            if np.any(np.isnan(sol['x'])) else sol['x']

        s = np.zeros_like(sol['s']) \
            if np.any(np.isnan(sol['s'])) else sol['s']

        y = np.zeros_like(sol['y']) \
            if np.any(np.isnan(sol['y'])) else sol['y']

        if np.allclose(y, 0.) and c@x < 0:
            obj = c@x
            # assert obj < 0
            x /= -obj
            s /= -obj
            # print('primal res:', np.linalg.norm(A@x + s))

        if np.allclose(s, 0.) and b@y < 0:
            obj = b@y
            # assert obj < 0
            y /= -obj
            # print('dual res:', np.linalg.norm(A.T@y))

        # print('SCS NONSOLVED')
        # print('x', x)
        # print('s', s)
        # print('y', y)

        z = xsy2z(x, s, y, tau=0., kappa=1.)

    return z, info


def ecos_solve(A, b, c, dim_dict, **kwargs):
    """Wraps ecos.solve for convenience."""

    ###
    # ECOS uses a different definition of the exp cone,
    # with y and z switched. In the future I might wrap it
    # (i.e., switch rows of A and elements of b, and switch
    # elements of the solutions s and y) but for now
    # I'm not supporting exp cones in ecos.
    ###

    ecos_cones = {'l': dim_dict['l'] if 'l' in dim_dict else 0,
                  'q': dim_dict['q'] if 'q' in dim_dict else []}  # ,
    # 'e': dim_dict['ep'] if 'ep' in dim_dict else 0}
    # print(ecos_cones)
    if ('ep' in dim_dict and dim_dict['ep'] > 0
            or 's' in dim_dict and len(dim_dict['s']) > 0):
        raise SolverError(
            'Only zero, linear, and second order cones supported.')
    zero = 0 if 'z' not in dim_dict else dim_dict['z']
    ecos_A, ecos_G = A[:zero, :], A[zero:, :]
    ecos_b, ecos_h = b[:zero], b[zero:]
    sol = ecos.solve(c=c, G=ecos_G, h=ecos_h, dims=ecos_cones,
                     A=ecos_A, b=ecos_b, **kwargs)

    solution = True

    x = sol['x']
    s = np.concatenate([np.zeros(zero), sol['s']])
    # not sure we can trust this
    # s = b - A@x
    y = np.concatenate([sol['y'], sol['z']])

    if sol['info']['exitFlag'] == 0:  # check that things make sense
        print('prim abs res.', np.linalg.norm(A@x + s - b))
        print('dua abs res.', np.linalg.norm(A.T@y + c))
        print('s^T y', s@y)

    if sol['info']['exitFlag'] in [1, 11]:  # infeas
        solution = False
        obj = b@y
        assert (obj < 0)
        y /= -obj

        print('primal infeas. cert residual norm', np.linalg.norm(A.T@y))
        #cones = dim2cones(dim_dict)
        proj = prod_cone.Pi(-y, *make_prod_cone_cache(dim_dict))
        print('primal infeas dist from cone', np.linalg.norm(proj))
        # if not (np.linalg.norm(proj) == 0.) and sol['info']['exitFlag'] == 1.:
        #     raise SolverError

        x = np.zeros_like(x)
        s = np.zeros_like(s)

    if sol['info']['exitFlag'] in [2, 12]:  # unbound
        solution = False
        obj = c@x
        assert (obj < 0)
        x /= -obj
        s /= -obj

        print('dual infeas. cert residual norm', np.linalg.norm(A@x + s))
        proj = prod_cone.Pi(s, *make_prod_cone_cache(dim_dict))
        print('dual infeas cert dist from cone', np.linalg.norm(s - proj))
        # if not (np.linalg.norm(s - proj) == 0.) and sol['info']['exitFlag'] == 2.:
        #     raise SolverError
        y = np.zeros_like(y)

    # print('ECOS SOLUTION')
    # print('solution', solution)
    # print('x', x)
    # print('s', s)
    # print('y', y)

    z = xsy2z(x, s, y, tau=solution, kappa=not solution)

    return z, sol['info']


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
    A = sp.csc_matrix(A)
    #A_tr = sp.csc_matrix(A.T)
    new_residual, u, v = residual_and_uv(
        z, (A.indptr, A.indices, A.data), b, c, make_prod_cone_cache(dim_dict))
    x, s, y, tau, kappa = uv2xsytaukappa(u, v, A.shape[1])

    pres = np.linalg.norm(A@x + s - b) / (1 + np.linalg.norm(b))
    dres = np.linalg.norm(A.T@y + c) / (1 + np.linalg.norm(c))
    gap = np.abs(c@x + b@y) / (1 + np.abs(c@x) + np.abs(b@y))

    print('pres %.2e, dres %.2e, gap %.2e' % (pres, dres, gap))

    z_plus = refine(A, b, c, dim_dict, z,
                    verbose=verbose,
                    iters=max_iters,
                    lsqr_iters=max_lsqr_iters)  # ,
    # max_runtime=solver_time * refine_solver_time_ratio)

    if return_z:
        return z_plus, info
    else:
        new_residual, u, v =\
            residual_and_uv(z_plus, (A.indptr, A.indices, A.data), b, c,
                            make_prod_cone_cache(dim_dict))
        x, s, y, tau, kappa = uv2xsytaukappa(u, v, A.shape[1])
        pres = np.linalg.norm(A@x + s - b) / (1 + np.linalg.norm(b))
        dres = np.linalg.norm(A.T@y + c) / (1 + np.linalg.norm(c))
        gap = np.abs(c@x + b@y) / (1 + np.abs(c@x) + np.abs(b@y))
        print('pres %.2e, dres %.2e, gap %.2e' % (pres, dres, gap))
        return x, s, y, info
