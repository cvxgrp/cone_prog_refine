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

HOW_MANY_DERIVATIVE_SAMPLES = 1000
HOW_LONG_DERIVATIVE_TEST_STEP = 1e-5


def size_vec(x):
    return 1 if isinstance(x, float) else len(x)


class MiscTest(unittest.TestCase):
    """ Test functions in the misc class."""

    def test_mat2vec(self):
        self.assertTrue(np.alltrue(mat2vec(np.eye(2)) == [1, 0, 1]))
        self.assertTrue(np.alltrue(
            mat2vec(np.array([[1, -1.], [-1, 1]])) == [1, -np.sqrt(2), 1]))
        self.assertTrue(np.alltrue(
            mat2vec(np.array([[1, -1, 0.], [-1, 1, 0], [0, 0, 1]])) ==
            [1, -np.sqrt(2), 0, 1, 0, 1]))

    def test_vec2mat(self):
        self.assertTrue(np.alltrue(vec2mat(np.array([1, 0., 1])) == np.eye(2)))
        self.assertTrue(np.alltrue(
            vec2mat(np.array([1, -np.sqrt(2), 1.])) == np.array([[1., -1], [-1, 1]])))
        self.assertTrue(np.alltrue(
            vec2mat(np.array([1., -np.sqrt(2), 0, 1, 0, 1])) ==
            np.array([[1, -1, 0], [-1, 1, 0], [0, 0, 1]])))


class BaseTestCone(unittest.TestCase):

    """Base class for cones tests."""

    sample_vecs = []
    sample_vec_proj = []
    sample_vecs_are_in = []
    sample_vecs_are_diff = []

    def make_cache(self, n):
        if not self.test_cone is semi_def_cone_single_cache:
            return np.empty(n)
        m = sizevec2sizemat(n)
        return (np.empty(m**2), np.empty(m))

    def test_contains(self):
        for x, isin in zip(self.sample_vecs, self.sample_vecs_are_in):
            cache = self.make_cache(len(x))
            res = self.test_cone.Pi(x, cache)
            Pix = res
            self.assertTrue(np.alltrue(Pix == x) == isin)

    def test_proj(self):
        for x, proj_x in zip(self.sample_vecs, self.sample_vec_proj):
            cache = self.make_cache(len(x))
            Pix = self.test_cone.Pi(x, cache)
            self.assertTrue(np.allclose(Pix, proj_x))

    def test_derivative_random(self):
        for x, isdiff in zip(self.sample_vecs,
                             self.sample_vecs_are_diff):

            x = np.random.randn(len(x))
            cache = self.make_cache(len(x))

            proj_x = self.test_cone.Pi(x, cache)

            # if not isdiff:
            #     pass
            # delta = np.random.randn(size_vec(x)) * 0.0001
            # self.assertRaises(NonDifferentiable,
            #                   self.test_cone.D(x, delta, cache))

            # else:
            for i in range(HOW_MANY_DERIVATIVE_SAMPLES):

                delta = np.random.randn(
                    size_vec(x)) * HOW_LONG_DERIVATIVE_TEST_STEP
                # print('x + delta:', x + delta)
                new_cache = self.make_cache(len(x))
                proj_x_plus_delta = self.test_cone.Pi(x + delta, new_cache)
                # proj_x_plus_delta, new_cache = self.test_cone.Pi(x + delta)
                # print('x, delta, cache:', x, delta, cache)
                dproj_x = self.test_cone.D(x, delta, cache)

                if not np.allclose(proj_x + dproj_x, proj_x_plus_delta):
                    print('x:', x)
                    print('Pi x:', proj_x)
                    print('delta:')
                    print(delta)
                    print('Pi (x + delta) - Pi(x):')
                    print(proj_x_plus_delta - proj_x)
                    print('DPi delta:')
                    print(dproj_x)

                self.assertTrue(np.allclose(
                    proj_x + dproj_x,
                    proj_x_plus_delta))

                # self.assertTrue(np.allclose(
                #     proj_x + deriv.T@delta,
                #     proj_x_plus_delta))


class TestNonNeg(BaseTestCone):

    test_cone = non_neg_cone_cached
    sample_vecs = [np.array(el, dtype=float) for el in
                   [np.array([-1., 0., 1.]), [1.], [0.], [],
                    [-1.], [-2, 2.], [1, 1], np.arange(1, 100), [-2, -1, 1, 2]]]
    sample_vec_proj = [np.array(el) for el in
                       [np.array([0., 0., 1.]), [1.], [0.], [],
                        [0.], [0., 2.], [1, 1], np.arange(1, 100), [0, 0, 1, 2]]]
    sample_vecs_are_in = [False, True, True, True,
                          False, False, True, True, False]
    sample_vecs_are_diff = [False, True, False, True,
                            True, True, True, True, True]


class TestFree(BaseTestCone):

    test_cone = free_cone_cached
    sample_vecs = [np.array(el, dtype=float) for el in
                   [np.array([-1., 0., 1.]), [1.], [0.], [],
                    [-1.], [-2., 2.], [1, 1], np.arange(1, 100), [-2, -1, 1, 2]]]
    sample_vecs_proj = sample_vecs
    sample_vecs_are_in = [True] * len(sample_vecs)
    sample_vecs_are_diff = [True] * len(sample_vecs)


class TestZero(BaseTestCone):

    test_cone = zero_cone_cached
    sample_vecs = [np.array(el, dtype=float) for el in
                   [np.array([-1., 0., 1.]), [1.],
                    [-1.], [-2., 2.], [1,
                                       1], np.arange(1, 100), [-2, -1, 1, 2],
                    [0.], [0., 0.], np.zeros(10), np.array([])]]
    sample_vec_proj = [np.array(el) for el in
                       [np.array([0., 0., 0.]), [0.],
                        [0.], [0., 0.], [0, 0], np.zeros(99), [0, 0, 0, 0],
                        [0.], [0, 0], np.zeros(10), []]]
    sample_vecs_are_in = [False] * (len(sample_vecs) - 4) + [True] * 4
    sample_vecs_are_diff = [True] * len(sample_vecs)


class TestExpPri(BaseTestCone):

    test_cone = exp_pri_cone
    sample_vecs = [np.array([0., 0., 0.]),
                   np.array([-10., -10., -10.]),
                   np.array([10., 10., 10.]),
                   np.array([1., 2., 3.]),
                   np.array([100., 2., 300.]),
                   np.array([-1., -2., -3.]),
                   np.array([-10., -10., 10.]),
                   np.array([1., -1.,  1.]),
                   np.array([0.08755124, -1.22543552, 0.84436298])]
    sample_vec_proj = [np.array([0., 0., 0.]),
                       np.array([-10., 0., 0.]),
                       np.array([4.26306172,  7.51672777, 13.25366605]),
                       np.array([0.8899428, 1.94041882, 3.06957225]),
                       np.array([73.77502858,  33.51053837, 302.90131756]),
                       np.array([-1., 0., 0.]),
                       np.array([-10., 0., 10.]),
                       np.array([0.22972088, 0.09487128, 1.06839895]),
                       np.array([3.88378507e-06, 2.58963810e-07, 0.84436298])]
    sample_vecs_are_in = [True, False, False,
                          False, False, False, False, False]
    sample_vecs_are_diff = [False, True, True, True, True, True, True, True]


class TestExpDua(BaseTestCone):

    test_cone = exp_dua_cone
    sample_vecs = [np.array([0., 0., 0.]),
                   np.array([-1., 1., 100.]),
                   np.array([1., 1., 100.]),
                   np.array([-1., -2., -3.])]
    sample_vec_proj = [np.array([0., 0., 0.]),
                       np.array([-1., 1., 100.]),
                       np.array([0., 1., 100.]),
                       np.array([-0.1100572, -0.05958119,  0.06957226])]
    sample_vecs_are_in = [True, True, False, False]
    sample_vecs_are_diff = [False, True, True, True]


class TestSecondOrder(BaseTestCone):

    test_cone = sec_ord_cone
    sample_vecs = [np.array([1., 0., 0.]),
                   np.array([1., 2., 2.]),
                   np.array([-10., 2., 2.]),
                   np.array([-2 * np.sqrt(2), 2., 2.]),
                   np.array([-1., 2., 2.]),
                   np.array([0., 1.]),
                   np.array([.5, -.5])]
    sample_vec_proj = [np.array([1., 0., 0.]),
                       [(2 * np.sqrt(2) + 1) / (2),
                        (2 * np.sqrt(2) + 1) / (2 * np.sqrt(2)),
                        (2 * np.sqrt(2) + 1) / (2 * np.sqrt(2))],
                       np.array([0, 0, 0]),
                       np.array([0, 0, 0]),
                       np.array([0.9142135623730951,
                                 0.6464466094067263,
                                 0.6464466094067263
                                 ]),
                       np.array([.5, .5]),
                       np.array([.5, -.5])]
    sample_vecs_are_in = [True, False, False, False, False, False, True]
    sample_vecs_are_diff = [True, True, True, False, True, True, False]


class TestProduct(BaseTestCone):

    test_cone = prod_cone

    def test_baseProduct(self):

        cache = make_prod_cone_cache({'l': 3})
        Pix = prod_cone.Pi(np.arange(3.), *cache)
        self.assertTrue(np.alltrue(Pix == np.arange(3.)))

        cache = make_prod_cone_cache({'l': 3})
        Pix = prod_cone.Pi(np.array([1., -1., -1.]), *cache)
        self.assertTrue(np.alltrue(Pix == [1, 0, 0]))

        # cones = [[non_neg_cone, 2], [semi_def_cone, 1]]
        cache = make_prod_cone_cache({'l': 2, 's': [1]})
        Pix = prod_cone.Pi(np.arange(3.), *cache)
        self.assertTrue(np.alltrue(Pix == range(3)))
        Pix = prod_cone.Pi(np.array([1., -1., -1.]), *cache)
        self.assertTrue(np.alltrue(Pix == [1, 0, 0]))

        # cones = [[semi_def_cone, 3], [semi_def_cone, 1]]
        cache = make_prod_cone_cache({'s': [2, 1]})
        Pix = prod_cone.Pi(np.arange(4.), *cache)
        self.assertTrue(np.allclose(Pix - np.array(
            [0.20412415, 0.90824829, 2.02062073, 3]), 0.))
        Pix = prod_cone.Pi(np.array([1, -20., 1, -1]), *cache)

        self.assertTrue(np.allclose(Pix, np.array([7.57106781, -10.70710678,
                                                   7.57106781,   0.])))

    def test_deriv_Product(self):

        dim_dict = {'z': 2, 'l': 20,  'q': [3, 4],
                    's': [3, 4, 5], 'ep': 10, 'ed': 10}
        cache = make_prod_cone_cache(dim_dict)
        m = 22 + 7 + (6 + 10 + 15) + 30 + 30
        # samples = [np.array([-5.3, 2., 11]), np.random.randn(m)
        #            np.array([-10.3, -22., 13.])]

        for j in range(100):
            x = np.random.randn(m)
            proj_x = prod_cone.Pi(x, *cache)
            for i in range(HOW_MANY_DERIVATIVE_SAMPLES):
                delta = np.random.randn(size_vec(x)) * \
                    HOW_LONG_DERIVATIVE_TEST_STEP
                # print('x + delta:', x + delta)
                new_cache = make_prod_cone_cache(dim_dict)
                proj_x_plus_delta = prod_cone.Pi(x + delta, *new_cache)

                dproj_x = prod_cone.D(x, delta, *cache)

                if not np.allclose(proj_x + dproj_x, proj_x_plus_delta, atol=1e-6):
                    print(dim_dict)
                    print('x:', x)
                    print('Pi (x):', proj_x)
                    # print(delta)
                    print('Pi (x + delta) - Pi(x):')
                    print(proj_x_plus_delta - proj_x)
                    print('DPi delta:')
                    print(dproj_x)

                    print('error:')
                    print(proj_x + dproj_x -
                          proj_x_plus_delta)

                self.assertTrue(np.allclose(
                    proj_x + dproj_x,
                    proj_x_plus_delta, atol=1e-6))


class TestSemiDefinite(BaseTestCone):

    test_cone = semi_def_cone_single_cache
    sample_vecs = [np.array([2, 0, 0, 2, 0, 2.]), np.array([1.]), np.array([-1.]),
                   np.array([10, 20., 10.]),
                   np.array([10, 0., -3.]),
                   np.array([10, 20., 0., 10, 0., 10]),
                   np.array([1, 20., 30., 4, 50., 6]),
                   np.array([1, 20., 30., 4, 50., 6, 200., 20., 1., 0.])]
    sample_vec_proj = [[2, 0, 0, 2, 0, 2], [1.], [0.],
                       [[12.07106781, 17.07106781, 12.07106781]],
                       np.array([10, 0., 0.]),
                       np.array([12.07106781, 17.07106781,
                                 0., 12.07106781, 0., 10.]),
                       np.array([10.11931299, 19.85794691, 21.57712079,
                                 19.48442822, 29.94069045,
                                 23.00413782]),
                       np.array([10.52224268,  13.74405405,  21.782617,  10.28175521,
                                 99.29457317,   5.30953205, 117.32861549,  23.76075308,
                                 2.54829623,  69.3944742])]
    sample_vecs_are_in = [True, True, False, False, False, False, False, False]
    sample_vecs_are_diff = [True, True, True, True, True, True, True, True]


class TestEmbeddedCone(unittest.TestCase):

    def test_embedded_vars(self):
        dim_dict = {'z': 2, 'l': 3, 'q': [4], 's': [3]}
        A, b, c, _, x_true, s_true, y_true = generate_problem(dim_dict,
                                                              mode='solvable')
        m, n = A.shape
        cone_caches = make_prod_cone_cache(dim_dict)
        # problem = ConicProblem(A, b, c, cones)
        u_true, v_true = xsy2uv(x_true, s_true, y_true, 1., 0.)
        x, s, y, _, _ = uv2xsytaukappa(u_true, v_true, len(x_true))

        self.assertTrue(np.alltrue(x == x_true))
        self.assertTrue(np.alltrue(s == s_true))
        self.assertTrue(np.alltrue(y == y_true))

        u, v = xsy2uv(x, s, y, 1., 0.)

        self.assertTrue(np.alltrue(u == u_true))
        self.assertTrue(np.alltrue(v == v_true))

        print('u', u)
        print('v', v)
        z = u - v
        print('z', z)

        proj_u = embedded_cone_Pi(z, *cone_caches, n)
        proj_v = proj_u - z

        print(' u = Pi z', proj_u)
        print('v = Pi z - z', proj_v)
        self.assertTrue(np.allclose(proj_u.T@proj_v, 0.))
        self.assertTrue(np.allclose(proj_u - u, 0.))
        self.assertTrue(np.allclose(proj_v - v, 0.))
        self.assertTrue(np.allclose(proj_u - proj_v, z))

    def test_embedded_cone_der_proj(self):
        for j in range(100):
            dim_dict = {'f': 2, 'l': 20,  'q': [2, 3, 5],
                        's': [3, 4, 5], 'ep': 10, 'ed': 10}
            A, b, c, _, x_true, s_true, y_true = generate_problem(
                dim_dict, mode='solvable')
            m, n = A.shape
            cone_caches = make_prod_cone_cache(dim_dict)
            # problem = ConicProblem(A, b, c, cones)
            u_true, v_true = xsy2uv(x_true, s_true, y_true, 1., 0.)
            z_true = u_true - v_true

            for i in range(HOW_MANY_DERIVATIVE_SAMPLES):
                delta = np.random.randn(len(z_true)) * \
                    HOW_LONG_DERIVATIVE_TEST_STEP
                proj_u = embedded_cone_Pi(z_true, *cone_caches, n)
                proj_v = proj_u - z_true

                self.assertTrue(np.allclose(proj_u - u_true, 0.))
                self.assertTrue(np.allclose(proj_v - v_true, 0.))
                dproj = embedded_cone_D(z_true, delta, *cone_caches, n)

                # deriv = EmbeddedConeDerProj(problem.n, z_true, cone)
                new_cone_caches = make_prod_cone_cache(dim_dict)
                u_plus_delta = embedded_cone_Pi(
                    z_true + delta, *new_cone_caches, n)

                # u_plus_delta, v_plus_delta = problem.embedded_cone_proj(z_true + delta)
                # dproj = deriv@delta

                error = u_true + dproj - u_plus_delta
                m, n = A.shape
                if not np.allclose(error, 0., atol=1e-6):
                    print('z:', z_true)
                    print('delta:')
                    print(delta)
                    print('Pi (z + delta) - Pi(z):')
                    print(u_plus_delta - u_true)
                    print('DPi delta:')
                    print(dproj)
                    print('error:')
                    print(error[:n])
                    print(error[n:-1])
                    print(error[-1])

                self.assertTrue(np.allclose(
                    error, 0., atol=1e-6))


if __name__ == '__main__':
    unittest.main()
