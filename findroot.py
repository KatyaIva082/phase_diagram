import numpy as np


def findroot(z, K):
    # Инициализация параметров
    Fv_min = 1 / (1 - np.max(K))
    Fv_max = 1 / (1 - np.min(K))

    a = Fv_min + 0.00001
    b = Fv_max - 0.00001

    # Размерность входных данных
    n = np.shape(z)

    # Начальное значение X
    X = (a + b) / 2

    # Вычисления для начальных значений
    fa = np.sum(z * (K - 1) / (1 + a * (K - 1)))
    fb = np.sum(z * (K - 1) / (1 + b * (K - 1)))
    fX = np.sum(z * (K - 1) / (1 + X * (K - 1)))

    # Итерационный процесс
    while abs(a - b) > 0.0000001:
        fa = np.sum(z * (K - 1) / (1 + a * (K - 1)))
        fb = np.sum(z * (K - 1) / (1 + b * (K - 1)))
        fX = np.sum(z * (K - 1) / (1 + X * (K - 1)))

        if fa * fX < 0:
            b = X
        else:
            if fb * fX < 0:
                a = X

        X = (a + b) / 2

    return X