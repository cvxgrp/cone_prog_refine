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

import time
import numpy as np

from .refine import refine
from .problem import residual_and_uv, uv2xsytaukappa, xsy2z
from .cones import make_prod_cone_cache

__all__ = ['cvxpy_solve']


def cvxpy_scs_to_cpsr(data, sol=None):

    A, b, c, dims = data['A'], data['b'], data['c'], data['dims']

    if sol is None:
        z = np.zeros(len(b) + len(c) + 1)
        z[-1] = 1.
    else:
        z = xsy2z(sol['x'], sol['s'], sol['y'], tau=1., kappa=0.)

        if np.any(np.isnan(z)):  # certificate...

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

            z = xsy2z(x, s, y, tau=0., kappa=1.)

    dims_dict = {}
    if int(dims.nonpos):
        dims_dict['l'] = int(dims.nonpos)
    if int(dims.zero):
        dims_dict['z'] = int(dims.zero)
    if int(dims.exp):
        dims_dict['ep'] = int(dims.exp)
    if len(dims.soc):
        dims_dict['q'] = list([int(el) for el in dims.soc])
    if len(dims.psd):
        dims_dict['s'] = list([int(el) for el in dims.psd])

    return A, b, c, z, dims_dict


def cvxpy_solve(cvxpy_problem, iters=2, lsqr_iters=30,
                presolve=False, scs_opts={},
                verbose=True, warm_start=True):
    from cvxpy.reductions.solvers.solving_chain import construct_solving_chain

    solving_chain = construct_solving_chain(cvxpy_problem, solver='SCS')
    data, inverse_data = solving_chain.apply(cvxpy_problem)

    start = time.time()
    if presolve:
        scs_solution = solving_chain.solve_via_data(cvxpy_problem,
                                                    data=data,
                                                    warm_start=warm_start,
                                                    verbose=verbose,
                                                    solver_opts=scs_opts)

        A, b, c, z, dims = cvxpy_scs_to_cpsr(data, scs_solution)
    else:
        A, b, c, z, dims = cvxpy_scs_to_cpsr(data)
        scs_solution = {}
        if warm_start and 'CPSR' in cvxpy_problem._solver_cache:
            z = cvxpy_problem._solver_cache['CPSR']['z']
    prepare_time = time.time() - start

    # TODO change this

    start = time.time()
    refined_z = refine(A, b, c, dims, z, iters=iters,
                       lsqr_iters=lsqr_iters, verbose=verbose)
    cvxpy_problem._solver_cache['CPSR'] = {'z': refined_z}
    refine_time = time.time() - start

    new_residual, u, v = residual_and_uv(
        refined_z, (A.indptr, A.indices, A.data), b, c, make_prod_cone_cache(dims))

    scs_solution['x'], scs_solution['s'], scs_solution[
        'y'], tau, kappa = uv2xsytaukappa(u, v, A.shape[1])

    scs_solution["info"] = {'status': 'Solved', 'solveTime': refine_time,
                            'setupTime': prepare_time, 'iter': iters, 'pobj': scs_solution['x'] @ c if tau > 0 else np.nan}

    cvxpy_problem.unpack_results(scs_solution, solving_chain, inverse_data)

    return cvxpy_problem.value
