import numba as nb
import numpy as np


@nb.njit()
def fourth_case_D_new(x, y, z, x_star, y_star, z_star, dx, dy, dz):
    """From BMB'18 appendix C."""

    mu_star = z_star - z

    matrix = np.zeros((4, 4))

    if y_star == 0.:
        print('z', x, y, z)
        print('pi z', x_star, y_star, z_star)
        raise Exception("y_star = 0.")
        # return np.zeros(3)

    alpha = x_star / y_star
    beta = np.exp(alpha)
    gamma = mu_star * beta / y_star

    matrix[0, 0] = 1 + gamma
    matrix[0, 1] = - gamma * alpha
    matrix[0, 2] = 0
    matrix[0, 3] = beta

    matrix[1, 0] = matrix[0, 1]
    matrix[1, 1] = 1 + gamma * alpha * alpha
    matrix[1, 2] = 0
    matrix[1, 3] = (1 - alpha) * beta

    matrix[2, 0] = 0
    matrix[2, 1] = 0
    matrix[2, 2] = 1
    matrix[2, 3] = -1

    matrix[3, 0] = beta
    matrix[3, 1] = matrix[1, 3]
    matrix[3, 2] = -1
    matrix[3, 3] = 0

    matinv = np.linalg.inv(matrix)

    jacobian = matinv[:3, :3]

    d = np.empty(3)

    d[0], d[1], d[2] = dx, dy, dz

    return jacobian @ d


def isin_kexp(z):
    return (((z[1] * np.exp(z[0] / z[1]) - z[2] <= 0.) and (z[1] > 0)) or
            ((z[0] <= 0) and (z[1] == 0) and (z[2] >= 0)))


def isin_minus_kexp_star(z):
    r, s, t = z
    return (((-r < 0) and r * exp(s / r) + np.e * t <= 0) or
            ((r == 0.) and (-s >= 0) and (-t >= 0)))


def isin_special_case(z):
    return ((z[0] <= 0) and z[1] <= 0)


@nb.jit(nb.float64[:](nb.float64[:], nb.float64[:]), nopython=True)
def exp_pri_Pi(z, cache):

    if isin_kexp(z):
        cache[:] = z
        return z

    if isin_minus_kexp_star(z):
        z[:] = 0.
        cache[:] = z
        return z

    if isin_special_case(z):
        z[1] = 0.
        z[2] = max(z[2], 0.)
        cache[:] = z
        return z

    exp_fourth_case_proj(z)

    cache[:] = z
    return z
