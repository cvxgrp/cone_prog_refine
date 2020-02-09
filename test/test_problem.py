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

import unittest
import numpy as np

from cpr import *
import scipy.sparse as sp

import time


class ProblemTest(unittest.TestCase):

    def test_Q(self):
        dim_dict = {'f': 10, 'l': 20, 'q': [10], 's': [5], 'ep': 10, 'ed': 10}
        A, b, c, _, x_true, s_true, y_true = generate_problem(dim_dict,
                                                              mode='solvable')

        # cones = dim2cones(dim_dict)
        cone_caches = make_prod_cone_cache(dim_dict)
        u, v = xsy2uv(x_true, s_true, y_true)
        res = residual(u - v, A, b, c, cone_caches)

        self.assertTrue(np.allclose(res[:-1], 0.))
        self.assertTrue(np.allclose(res[-1], 0.))

    # def test_embedded_vars(self):
    #     dim_dict = {'f': 2, 'l': 3, 'q': [4], 's': [3]}
    #     A, b, c, _, x_true, s_true, y_true = generate_problem(dim_dict,
    #                                                           mode='solvable')
    #     m, n = A.shape
    #     cone_caches = make_prod_cone_cache(dim_dict)
    #     # problem = ConicProblem(A, b, c, cones)
    #     u_true, v_true = xsy2uv(x_true, s_true, y_true, 1., 0.)
    #     x, s, y, _, _ = uv2xsytaukappa(u_true, v_true, len(x_true))

    #     self.assertTrue(np.alltrue(x == x_true))
    #     self.assertTrue(np.alltrue(s == s_true))
    #     self.assertTrue(np.alltrue(y == y_true))

    #     u, v = xsy2uv(x, s, y, 1., 0.)

    #     self.assertTrue(np.alltrue(u == u_true))
    #     self.assertTrue(np.alltrue(v == v_true))

    #     print('u', u)
    #     print('v', v)
    #     z = u - v
    #     print('z', z)

    #     proj_u = embedded_cone_Pi(z, cone_caches, n)
    #     proj_v = proj_u - z

    #     print(' u = Pi z', proj_u)
    #     print('v = Pi z - z', proj_v)
    #     self.assertTrue(np.allclose(proj_u.T@proj_v, 0.))
    #     self.assertTrue(np.allclose(proj_u - u, 0.))
    #     self.assertTrue(np.allclose(proj_v - v, 0.))
    #     self.assertTrue(np.allclose(proj_u - proj_v, z))

    # def test_embedded_cone_der_proj(self):
    #     dim_dict = {'f': 2, 'l': 20, 'q': [2, 3, 5], 's': [3, 4], 'ep': 4}
    #     A, b, c, _, x_true, s_true, y_true = generate_problem(
    #         dim_dict, mode='solvable')
    #     m, n = A.shape
    #     cone_caches = make_prod_cone_cache(dim_dict)
    #     #problem = ConicProblem(A, b, c, cones)
    #     u_true, v_true = xsy2uv(x_true, s_true, y_true, 1., 0.)
    #     z_true = u_true - v_true

    #     delta = np.random.randn(len(z_true)) * 1E-7
    #     proj_u = embedded_cone_Pi(z_true, cone_caches, n)
    #     proj_v = proj_u - z_true

    #     self.assertTrue(np.allclose(proj_u - u_true, 0.))
    #     self.assertTrue(np.allclose(proj_v - v_true, 0.))
    #     dproj = embedded_cone_D(z_true, delta, cone_caches, n)

    #     #deriv = EmbeddedConeDerProj(problem.n, z_true, cone)
    #     new_cone_caches = make_prod_cone_cache(dim_dict)
    #     u_plus_delta = embedded_cone_Pi(
    #         z_true + delta, new_cone_caches, n)

    #     #u_plus_delta, v_plus_delta = problem.embedded_cone_proj(z_true + delta)
    #     # dproj = deriv@delta

    #     print('delta:')
    #     print(delta)
    #     print('Pi (z + delta) - Pi(z):')
    #     print(u_plus_delta - u_true)
    #     print('DPi delta:')
    #     print(dproj)
    #     print('error:')
    #     print(u_true + dproj - u_plus_delta)

    #     self.assertTrue(np.allclose(
    #         u_true + dproj,
    #         u_plus_delta, atol=1E-6))

    def test_residual_der(self):
        dim_dict = {'l': 10, 'q': [5, 10], 's': [3, 4], 'ep': 10, 'ed': 2}
        A, b, c, _, x_true, s_true, y_true = generate_problem(
            dim_dict, mode='solvable', density=.3)
        m, n = A.shape
        cones_caches = make_prod_cone_cache(dim_dict)
        u_true, v_true = xsy2uv(x_true, s_true, y_true, 1., 0.)
        z_true = u_true - v_true

        res = residual(z_true, A,  # A.T,
                       b, c, cones_caches)
        delta = np.random.randn(len(z_true)) * 1E-7
        residual_z_plus_delta = residual(z_true + delta, A,  # A.T,
                                         b, c,
                                         make_prod_cone_cache(dim_dict))

        A = sp.csc_matrix(A)
        # A_tr = sp.csc_matrix(A.T)

        (A.indptr, A.indices, A.data),
        #(A_tr.indptr, A_tr.indices, A_tr.data),

        dres = residual_D(z_true, delta, (A.indptr, A.indices, A.data),
                          #(A_tr.indptr, A_tr.indices, A_tr.data),
                          b, c, cones_caches)

        print('delta:')
        print(delta)
        print('Res (z + delta) - Res(z):')
        print(residual_z_plus_delta - res)
        print('dRes')
        print(dres)

        self.assertTrue(np.allclose(
            res + dres,
            residual_z_plus_delta, atol=1e-5))

        # print('testing DT')
        # res, cones_caches = residual(z_true, A, b, c, cones)
        # delta = np.random.randn(len(z_true)) * 1E-5
        # #residual_z_plus_delta, _ = residual(z_true + delta, A, b, c, cones)
        # dz = residual_DT(z_true, delta, A, b, c, cones_caches)
        # new_res, _ = residual(z_true + dz, A, b, c, cones)

        # print('delta:')
        # print(delta)
        # print('dz')
        # print(dz)
        # print('dRes')
        # print(new_res - res)

        # self.assertTrue(np.allclose(
        #     new_res - res,
        #     delta))

    def check_refine_ecos(self, dim_dict, **kwargs):
        solvable = True
        if not ('mode' in kwargs):
            kwargs['mode'] = 'solvable'
        if (kwargs['mode'] != 'solvable'):
            solvable = False
        print('generating problem')
        A, b, c, dim_dict, x_true, s_true, y_true = generate_problem(dim_dict,
                                                                     **kwargs)
        m, n = A.shape

        u, v = xsy2uv(x_true, s_true, y_true, solvable, not solvable)

        embedded_res = residual(u - v, A, b, c,
                                make_prod_cone_cache(dim_dict))
        self.assertTrue(np.allclose(embedded_res, 0.))

        print('calling solver')
        solver_start = time.time()
        z, info = ecos_solve(A, b, c, dim_dict,
                             verbose=True,
                             feastol=1e-15,
                             reltol=1e-15,
                             abstol=1e-15,
                             )
        solver_end = time.time()
        pridua_res = residual(z, A, b, c, make_prod_cone_cache(dim_dict))
        if not (np.alltrue(pridua_res == 0.)):
            refine_start = time.time()
            z_plus = refine(A, b, c, dim_dict, z)
            refine_end = time.time()

            pridua_res_new = residual(
                z_plus, A, b, c, make_prod_cone_cache(dim_dict))
            print('\n\nSolver time: %.2e' % (solver_end - solver_start))
            print("||pridua_res before refinement||")
            oldnorm = np.linalg.norm(pridua_res)
            print('%.6e' % oldnorm)
            print('\n\nRefinement time: %.2e' % (refine_end - refine_start))
            print("||pridua_res after refinement||")
            newnorm = np.linalg.norm(pridua_res_new)
            print('%.6e' % newnorm)
            self.assertTrue(newnorm <= oldnorm)
            if (newnorm == oldnorm):
                print('\n\n\nrefinement FAILED!!!, dims %s\n\n\n' % dim_dict)
            else:
                print('\n\n\nrefinement SUCCEDED!!!, dims %s\n\n\n' % dim_dict)
                print('new optval', c@z_plus[:n])

    def check_refine_scs(self, dim_dict, **kwargs):
        solvable = True
        if not ('mode' in kwargs):
            kwargs['mode'] = 'solvable'
        if (kwargs['mode'] != 'solvable'):
            solvable = False
        print('generating problem')
        A, b, c, _, x_true, s_true, y_true = generate_problem(
            dim_dict, **kwargs)
        m, n = A.shape

        u, v = xsy2uv(x_true, s_true, y_true, solvable, not solvable)

        embedded_res = residual(
            u - v, A, b, c, make_prod_cone_cache(dim_dict))
        self.assertTrue(np.allclose(embedded_res, 0.))

        print('calling solver')
        solver_start = time.time()
        z, info = scs_solve(A, b, c, dim_dict,
                            verbose=True,
                            eps=1e-15,
                            max_iters=1000)
        solver_end = time.time()
        pridua_res = residual(z, A, b, c, make_prod_cone_cache(dim_dict))
        if not (np.alltrue(pridua_res == 0.)):
            refine_start = time.time()
            z_plus = refine(A, b, c, dim_dict, z)
            refine_end = time.time()

            pridua_res_new = residual(
                z_plus, A, b, c, make_prod_cone_cache(dim_dict))
            print('\n\nSolver time: %.2e' % (solver_end - solver_start))
            print("||pridua_res before refinement||")
            oldnorm = np.linalg.norm(pridua_res)
            print('%.6e' % oldnorm)
            print('\n\nRefinement time: %.2e' % (refine_end - refine_start))
            print("||pridua_res after refinement||")
            newnorm = np.linalg.norm(pridua_res_new)
            print('%.6e' % newnorm)
            self.assertTrue(newnorm <= oldnorm)
            if (newnorm == oldnorm):
                print('\n\n\nrefinement FAILED!!!, dims %s\n\n\n' % dim_dict)
            else:
                print('\n\n\nrefinement SUCCEDED!!!, dims %s\n\n\n' % dim_dict)
                print('new optval', c@z_plus[:n])

    def test_solve_and_refine(self):
        for dims in [{'l': 20, 'q': [10] * 5},
                     {'l': 20, 's': [10] * 5},
                     #{'l': 290, 'q': [10] * 10},
                     #{'l': 1000},
                     {'l': 50, 'q': [10] * 5, 's':[20] * 1}]:
            np.random.seed(1)
            A, b, c, _, x_true, s_true, y_true = generate_problem(
                dims, mode='solvable')
            m, n = A.shape

            self.assertTrue(np.allclose(A@x_true + s_true - b, 0))
            self.assertTrue(np.allclose(A.T@y_true + c, 0))
            self.assertTrue(np.allclose(b.T@y_true + c.T@x_true, 0))

            x, s, y, info = solve(A, b, c, dims,
                                  solver='scs',
                                  solver_options={  # 'max_iters': 500,
                                      'eps': 1e-9,
                                      'verbose': True},
                                  refine_solver_time_ratio=5,
                                  verbose=True)
            self.assertTrue(np.allclose(A@x + s - b, 0, atol=1e-5))
            self.assertTrue(np.allclose(A.T@y + c, 0, atol=1e-5))
            self.assertTrue(np.allclose(b.T@y + c.T@x, 0, atol=1e-5))

    def test_infeasible(self):
        self.check_refine_scs({'l': 20, 'q': [10] * 5}, mode='infeasible')
        self.check_refine_ecos({'l': 20, 'q': [10] * 5}, mode='infeasible')
        # self.check_refine_scs({'s': [20]}, mode='infeasible')
        self.check_refine_scs({'s': [10]}, mode='infeasible')
        self.check_refine_scs({'q': [50]}, mode='infeasible')
        self.check_refine_ecos({'q': [50]}, mode='infeasible')

    def test_unbound(self):
        self.check_refine_scs({'l': 20, 'q': [10] * 5}, mode='unbounded')
        self.check_refine_ecos({'l': 20, 'q': [10] * 5}, mode='unbounded')
        # self.check_refine_scs({'s': [20]}, mode='unbounded')
        self.check_refine_scs({'s': [10], 'q': [5]}, mode='unbounded')
        self.check_refine_scs({'q': [50, 5]}, mode='unbounded')
        self.check_refine_ecos({'q': [50, 5]}, mode='unbounded')

    # def test_nondiff(self):
    #     self.check_refine_scs({'l': 20, 'q': [10] * 5}, nondiff_point=True)
    #    # self.check_refine_scs({'s': [20]}, nondiff_point=True)
    #     self.check_refine_scs({'s': [10]}, nondiff_point=True)
    #     self.check_refine_scs({'q': [50]}, nondiff_point=True)

    def test_scs(self):
        self.check_refine_scs({'l': 20, 'q': [10] * 5})
        self.check_refine_scs({'l': 20, 'q': [10] * 5, 'ep': 20})
        self.check_refine_scs({'l': 20, 'q': [10] * 5, 'ed': 20})

        # self.check_refine_scs({'s': [20]})
        # self.check_refine_scs({'s': [10]})
        self.check_refine_scs({'q': [50]})
        self.check_refine_scs({'l': 10})
        # self.check_refine_scs({'l': 50})
        self.check_refine_scs({'l': 20, 'q': [10, 20]})
        # self.check_refine_scs({'l': 1000, 'q': [100] * 10})
        # self.check_refine_scs({'f': 10, 'l': 20, 'q': [10], 's': [10]})
        self.check_refine_scs({'f': 10, 'l': 20, 'q': [10, 20], 's': [5, 10]})

    def test_ecos(self):
        self.check_refine_ecos({'l': 20, 'q': [10] * 5})
        # self.check_refine_ecos({'l': 50, 'q': [10] * 10})
        self.check_refine_ecos({'q': [50]})
        self.check_refine_ecos({'l': 10})
        # self.check_refine_ecos({'l': 50})
        # self.check_refine_ecos({'l': 1000})
        self.check_refine_ecos({'l': 20, 'q': [10, 20, 40]})
        # self.check_refine_ecos({'l': 20, 'q': [10, 20, 40, 60]})


class CVXPYTest(unittest.TestCase):

    def test_CVXPY(self):

        try:
            import cvxpy as cvx
        except ImportError:
            return

        m, n = 40, 30
        A = np.random.randn(m, n)
        b = np.random.randn(m)
        x = cvx.Variable(n)

        p = cvx.Problem(cvx.Minimize(cvx.sum_squares(x)),
                        [cvx.log_sum_exp(x) <= 10, A @ x <= b])

        cvxpy_solve(p, presolve=True, iters=10, scs_opts={'eps': 1E-10})

        self.assertTrue(np.alltrue(A @ x.value - b <= 1E-8))
