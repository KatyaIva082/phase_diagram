import numpy as np


# Функция для решения кубического уравнения
def cubic_equation_solver(a, b, c, d):
    inv_a = 1. / a
    b_a = inv_a * b
    b_a2 = b_a * b_a
    c_a = inv_a * c
    d_a = inv_a * d

    Q = (3 * c_a - b_a2) / 9
    R = (9 * b_a * c_a - 27 * d_a - 2 * b_a * b_a2) / 54
    Q3 = Q * Q * Q
    D = Q3 + R * R
    b_a_3 = (1. / 3.) * b_a

    if Q == 0:
        if R == 0:
            x0 = x1 = x2 = - b_a_3
            return np.array([x0, x1, x2])
        else:
            cube_root = (2 * R) ** (1. / 3.)
            x0 = cube_root - b_a_3
            return np.array([x0])

    if D <= 0:
        theta = np.arccos(R / np.sqrt(-Q3))
        sqrt_Q = np.sqrt(-Q)
        x0 = 2 * sqrt_Q * np.cos(theta / 3.0) - b_a_3
        x1 = 2 * sqrt_Q * np.cos((theta + 2 * np.pi) / 3.0) - b_a_3
        x2 = 2 * sqrt_Q * np.cos((theta + 4 * np.pi) / 3.0) - b_a_3
        return np.array([x0, x1, x2])

    AD = 0.
    BD = 0.
    R_abs = np.fabs(R)
    if R_abs > 2.2204460492503131e-16:
        AD = (R_abs + np.sqrt(D)) ** (1. / 3.)
        AD = AD if R >= 0 else -AD
        BD = -Q / AD

    x0 = AD + BD - b_a_3
    return np.array([x0])
