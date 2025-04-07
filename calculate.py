import numpy as np
from cubic_solve import cubic_equation_solver
from Constants import critical_constants

def calculate_z(R, Tkr, Pkr, si, Wi, T, P, z, N, isMax):

    ac_i = 0.42747 * R ** 2 * Tkr ** 2 / Pkr
    psi_i = 0.48 + 1.574 * Wi - 0.176 * Wi ** 2
    alpha_i = (1 + psi_i * (1 - (T / Tkr) ** 0.5)) ** 2
    a_i = ac_i * alpha_i
    b_i = 0.08664 * R * Tkr / Pkr
    c_i = si

    # Суммирование для смеси
    aw = 0
    bw = 0
    cw = 0
    interaction_coefficient = 0
    for i in range(N):
        for j in range(N):
            # Вместо нуля нужно использовать коэффициент парного взаимодействия
            aw += z[i] * z[j] * (1 - interaction_coefficient) * np.sqrt(a_i[i] * a_i[j])

    bw = np.dot(z, b_i)
    cw = 0
    #cw = np.dot(z, c_i)

    # Формула для расчета
    Aw = aw * P / (R ** 2 * T ** 2)
    Bw = bw * P / (R * T)
    Cw = cw * P / (R * T)
    Biw = b_i * P / (R * T)
    Ciw = c_i * P / (R * T)

    coefficients = [1, 3 * Cw - 1, 3 * Cw ** 2 - Bw ** 2 - 2 * Cw - Bw + Aw,
                    Cw ** 3 - Bw ** 2 * Cw - Cw ** 2 - Bw * Cw + Aw * Cw - Aw * Bw]
    cubroot = cubic_equation_solver(coefficients[0], coefficients[1], coefficients[2], coefficients[3])

    Z = np.max(cubroot) if isMax else np.min(cubroot)

    interaction_coefficient = 0
    avvv = np.zeros(N)
    for i in range(N):
        avv = 0
        for j in range(N):
            avv += z[j] * (1 - interaction_coefficient) * (a_i[i] * a_i[j]) ** 0.5
        avvv[i] = avv


    # SRK-Peneloux
    fz_i = np.exp(np.log(z * P) - np.log(Z + Cw - Bw) + (Biw - Ciw) / (Z + Cw - Bw) -
                  (Aw / Bw) * ((2 * avvv / aw) - (b_i / bw)) * np.log((Z + Bw + Cw) / (Z + Cw)) -
                  (Aw / Bw) * (Biw + Ciw) / (Z + Bw + Cw) + (Aw / Bw) * Ciw / (Z + Cw))


    return fz_i, Z