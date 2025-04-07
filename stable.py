import numpy as np
import matplotlib.pyplot as plt
from Constants import critical_constants
from calculate import calculate_z

user_values = {
    'C1': 0.90,
    'C3': 0.05,
    'nC5': 0.05

}

# Определим диапазоны температур и давлений
T_min, T_max = 150, 400  # Минимальная и максимальная температура
P_min, P_max = 1, 20  # Минимальное и максимальное давление
T_step = 1  # Шаг по температуре
P_step = 0.1  # Шаг по давлению

# Создаем массивы для хранения результатов
T_range = np.arange(T_min, T_max + T_step, T_step)
P_range = np.arange(P_min, P_max + P_step, P_step)
stability_map = np.zeros((len(P_range), len(T_range)))  # Матрица стабильности

# Функция для проверки стабильности
def check_stability(T, P, user_values, critical_constants):
    R = 0.00831675
    N = len(user_values)
    z = np.array(list(user_values.values()))
    z = z / np.sum(z)

    Tkr = []
    Pkr = []
    Wi = []
    si = []
    Vkr = []
    Mwi = []
    Cp1_id = []
    Cp2_id = []
    Cp3_id = []
    Cp4_id = []

    for component in user_values:
        if component in critical_constants:
            Tkr.append(critical_constants[component]['Tkr'])
            Pkr.append(critical_constants[component]['Pkr'])
            Wi.append(critical_constants[component]['Wi'])
            si.append(critical_constants[component]['si'])
            Vkr.append(critical_constants[component]['Vkr'])
            Mwi.append(critical_constants[component]['Mwi'])
            Cp1_id.append(critical_constants[component]['Cp1_id'])
            Cp2_id.append(critical_constants[component]['Cp2_id'])
            Cp3_id.append(critical_constants[component]['Cp3_id'])
            Cp4_id.append(critical_constants[component]['Cp4_id'])
        else:
            Tkr.append(0)
            Pkr.append(0)
            Wi.append(0)
            si.append(0)
            Vkr.append(0)
            Mwi.append(0)
            Cp1_id.append(0)
            Cp2_id.append(0)
            Cp3_id.append(0)
            Cp4_id.append(0)

    Tkr = np.array(Tkr)
    Pkr = np.array(Pkr)
    Wi = np.array(Wi)
    si = np.array(si)
    Vkr = np.array(Vkr)
    Mwi = np.array(Mwi)
    Cp1_id = np.array(Cp1_id)
    Cp2_id = np.array(Cp2_id)
    Cp3_id = np.array(Cp3_id)
    Cp4_id = np.array(Cp4_id)

    K_i = (np.exp(5.373 * (1 + Wi) * (1 - Tkr / T)) * Pkr / P) ** 1.0

    fz_i, Z_init = calculate_z(R, Tkr, Pkr, si, Wi, T, P, z, N, 1)

    m = 0
    Ri_v = 1
    TS_v_flag = 0

    while m < 30:
        Yi_v = z * K_i
        Sv = np.sum(Yi_v)
        y_i = Yi_v / Sv

        fw_i, Z_l = calculate_z(R, Tkr, Pkr, si, Wi, T, P, y_i, N, 1)

        Ri = fz_i / (Sv * fw_i)
        Ri_v = np.sum((Ri - 1) ** 2)

        if Ri_v < 10 ** (-12):
            m = 30

        K_i = K_i * Ri
        TS_v = np.sum(np.log(K_i) ** 2)

        if TS_v < 10 ** (-4):
            TS_v_flag = 1
            m = 30

        m += 1

    K_iv = K_i

    K_i = (np.exp(5.373 * (1 + Wi) * (1 - Tkr / T)) * Pkr / P) ** 1.0

    fz_i, Z_init = calculate_z(R, Tkr, Pkr, si, Wi, T, P, z, N, 1)

    ml = 0
    Ri_l = 1
    TS_l_flag = 0

    while ml < 30:
        Yi_l = z / K_i
        Sl = np.sum(Yi_l)
        x_i = Yi_l / Sl

        fl_i, Z_l = calculate_z(R, Tkr, Pkr, si, Wi, T, P, x_i, N, 0)

        Ri = Sl * fl_i / fz_i
        Ri_l = np.sum((Ri - 1) ** 2)

        if Ri_l < 1e-12:
            ml = 30

        K_i = K_i * Ri
        TS = np.sum(np.log(K_i) ** 2)

        if TS < 1e-4:
            TS_l_flag = 1
            ml = 30

        ml += 1

    K_il = K_i

    if ((TS_l_flag == 1 and TS_v_flag == 1) or
        (Sv <= 1 and TS_l_flag == 1) or
        (Sl <= 1 and TS_v_flag == 1) or
        (Sv < 1 and Sl <= 1)):
        return 1  # Стабильное состояние
    else:
        return 0  # Нестабильное состояние

# Заполняем матрицу стабильности
for i, P in enumerate(P_range):
    for j, T in enumerate(T_range):
        stability_map[i, j] = check_stability(T, P, user_values, critical_constants)

# Визуализация
plt.figure(figsize=(10, 6))
plt.contourf(T_range, P_range, stability_map, levels=[-0.5, 0.5, 1.5], colors=['red', 'blue'], alpha=0.5)
plt.colorbar(label='Стабильность (1 - стабильно, 0 - нестабильно)')
plt.ylabel('Давление (МПа)')  # Теперь давление по оси Y
plt.xlabel('Температура (К)')  # Теперь температура по оси X
plt.title('Фазовая диаграмма')
plt.grid(True)
plt.show()