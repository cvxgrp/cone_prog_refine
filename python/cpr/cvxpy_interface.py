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

import time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator

from .refine import refine
from .problem import residual_and_uv, uv2xsytaukappa, xsy2z
from .cones import make_prod_cone_cache

__all__ = ['cvxpy_solve', 'cvxpy_differentiate']


def cvxpy_scs_to_cpr(data, sol=None):

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

        A, b, c, z, dims = cvxpy_scs_to_cpr(data, scs_solution)
    else:
        A, b, c, z, dims = cvxpy_scs_to_cpr(data)
        scs_solution = {}
        if warm_start and 'CPR' in cvxpy_problem._solver_cache:
            z = cvxpy_problem._solver_cache['CPR']['z']
    prepare_time = time.time() - start

    # TODO change this

    start = time.time()
    refined_z = refine(A, b, c, dims, z, iters=iters,
                       lsqr_iters=lsqr_iters, verbose=verbose)
    cvxpy_problem._solver_cache['CPR'] = {'z': refined_z}
    refine_time = time.time() - start

    new_residual, u, v = residual_and_uv(
        refined_z, (A.indptr, A.indices, A.data), b, c, make_prod_cone_cache(dims))

    scs_solution['x'], scs_solution['s'], scs_solution[
        'y'], tau, kappa = uv2xsytaukappa(u, v, A.shape[1])

    scs_solution["info"] = {'status': 'Solved', 'solveTime': refine_time,
                            'setupTime': prepare_time, 'iter': iters, 'pobj': scs_solution['x'] @ c if tau > 0 else np.nan}

    cvxpy_problem.unpack_results(scs_solution, solving_chain, inverse_data)

    return cvxpy_problem.value


class NotAffine(Exception):
    pass


def cvxpy_differentiate(cvxpy_problem, parameters, output_expression,
                        iters=2, lsqr_iters=30, derive_lsqr_iters=100,
                        presolve=False, scs_opts={},
                        verbose=True, warm_start=True):
    """Compute the derivative matrix of the solution map of
    a CVXPY problem, whose input is a list of CVXPY parameters,
    and output is a CVXPY one-dimensional expression of the solution,
    the constraint violations, or the dual parameters.

    Only affine CVXPY transformations are allowed.
    """

    solving_chain = construct_solving_chain(cvxpy_problem, solver='SCS')
    data, inverse_data = solving_chain.apply(cvxpy_problem)
    A, b, c, _, _ = cvxpy_scs_to_cpr(data)

    # A is a sparse matrix, so below we compute
    # sparse matrix differences.

    input_mappings = []

    # make mapping from input parameters to data
    for parameter in parameters:
        if verbose:
            print('compiling parameter', parameter.name())
        old_par_val = parameter.value
        parameter.value += 1.
        newdata, _ = solving_chain.apply(cvxpy_problem)
        new_A, new_b, new_c, _, _ = cvxpy_scs_to_cpr(newdata)
        dA = new_A - A
        db = new_b - b
        dc = new_c - c
        parameter.value -= 2.
        newdata, _ = solving_chain.apply(cvxpy_problem)
        new_A, new_b, new_c, _, _ = cvxpy_scs_to_cpr(newdata)
        if not (np.allclose(A - new_A, dA)) \
                and (np.allclose(b - new_b, db))\
                and (np.allclose(c - new_c, dc)):
            raise NotAffine('on parameter %s' % parameter.name())
        parameter.value = old_par_val
        input_mappings.append((dA, db, dc))

    _ = cvxpy_solve(cvxpy_problem, iters=iters, lsqr_iters=lsqr_iters,
                    presolve=presolve, scs_opts=scs_opts,
                    verbose=verbose, warm_start=warm_start)

    # used by cvxpy to transform back
    scs_solution = {}
    scs_solution["info"] = {'status': 'Solved', 'solveTime': 0.,
                            'setupTime': 0., 'iter': 0, 'pobj': np.nan}

    z = cvxpy_problem._solver_cache['CPR']['z']

    if not (len(output_expression.shape) == 1):
        raise ValueError('Only one-dimensional outputs')
    output_matrix = np.empty((len(z), output_expression.size))
    base = output_expression.value

    # make mapping from z to output
    for i in range(len(z)):
        # perturb z
        old_val = z[i]
        z[i] += old_val * 1e-8
        _, new_u, new_v = residual_and_uv(
            z, (A.indptr, A.indices, A.data), b, c,
            make_prod_cone_cache(dims))
        scs_solution['x'], scs_solution['s'], scs_solution[
            'y'], tau, kappa = uv2xsytaukappa(new_u, new_v, A.shape[1])
        cvxpy_problem.unpack_results(scs_solution, solving_chain,
                                     inverse_data)

        output_matrix[i, :] = output_expression.value - base

        z[i] -= old_val * 2e-8
        _, new_u, new_v = residual_and_uv(
            z, (A.indptr, A.indices, A.data), b, c,
            make_prod_cone_cache(dims))
        scs_solution['x'], scs_solution['s'], scs_solution[
            'y'], tau, kappa = uv2xsytaukappa(new_u, new_v, A.shape[1])
        cvxpy_problem.unpack_results(scs_solution, solving_chain,
                                     inverse_data)

        if not np.allclose(output_matrix[i, :],
                           base - output_expression.value):
            raise NotAffine('on solution variable z[%d]' % i)

        z[i] = old_val

    def matvec(d_parameters):
        assert len(d_parameters) == len(parameters)
        total_dA = sp.csc_matrix()
        total_db = np.zeros(len(b))
        total_dc = np.zeros(len(c))

        for i in len(d_parameters):
            total_dA += input_mappings[i][0] * d_parameters[i]
            total_db += input_mappings[i][1] * d_parameters[i]
            total_dc += input_mappings[i][2] * d_parameters[i]

        refined_z = refine(A + total_dA, b + total_db,
                           c + total_dc, dims, z, iters=1,
                           lsqr_iters=derive_lsqr_iters, verbose=verbose)
        dz = refined_z - z
        return dz @ output_matrix

    def rmatvec(d_output):
        dz = output_matrix @ d_output

    result = LinearOperator((len(parameters), output_expression.size),
                            matvec=matvec,
                            rmatvec=rmatvec,
                            dtype=np.float)
