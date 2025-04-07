import time
import numpy as np
import matplotlib.pyplot as plt
from Constants import critical_constants
from calculate import calculate_z
from findroot import findroot
from cubic_solve import cubic_equation_solver


def calculate_single_point(T, P, user_values, print_output=True):
    """Исходный код расчета для одной точки (T,P)"""
    start_time = time.time()
    R = 0.00831675

    # Получаем параметры для каждого компонента
    N = len(user_values)
    z = np.array(list(user_values.values()))  # Молярные доли компонентов
    z = z / np.sum(z)

    Tkr, Pkr, Wi, si, Vkr, Mwi = [], [], [], [], [], []
    Cp1_id, Cp2_id, Cp3_id, Cp4_id = [], [], [], []

    for component in user_values:
        if component in critical_constants:
            crit = critical_constants[component]
            Tkr.append(crit['Tkr'])
            Pkr.append(crit['Pkr'])
            Wi.append(crit['Wi'])
            si.append(crit['si'])
            Vkr.append(crit['Vkr'])
            Mwi.append(crit['Mwi'])
            Cp1_id.append(crit['Cp1_id'])
            Cp2_id.append(crit['Cp2_id'])
            Cp3_id.append(crit['Cp3_id'])
            Cp4_id.append(crit['Cp4_id'])
        else:
            if print_output:
                print(f"Предупреждение: данные для компонента {component} не найдены в critical_constants.")
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

    Tkr, Pkr, Wi, si, Vkr = map(np.array, [Tkr, Pkr, Wi, si, Vkr])
    V_pkr = np.dot(z, Vkr)
    T_pkr = np.dot(z, Tkr)
    Control_phase = V_pkr * T_pkr * T_pkr

    # Расчет стабильности
    K_i = (np.exp(5.373 * (1 + Wi) * (1 - Tkr / T)) * Pkr / P) ** 1.0
    fz_i, Z_init = calculate_z(R, Tkr, Pkr, si, Wi, T, P, z, N, 1)

    # Проверка газовой фазы
    m, TS_v_flag = 0, 0
    while m < 30:
        Yi_v = z * K_i
        Sv = np.sum(Yi_v)
        y_i = Yi_v / Sv
        fw_i, Z_l = calculate_z(R, Tkr, Pkr, si, Wi, T, P, y_i, N, 1)
        Ri = fz_i / (Sv * fw_i)
        Ri_v = np.sum((Ri - 1) ** 2)

        if Ri_v < 1e-12:
            if print_output:
                print('Сходимость достигнута по газу!!!')
            m = 30

        K_i = K_i * Ri
        TS_v = np.sum(np.log(K_i) ** 2)

        if TS_v < 1e-4:
            if print_output:
                print('TS по газу найдено!!!')
            TS_v_flag = 1
            m = 30

        m += 1

    K_iv = K_i

    # Проверка жидкой фазы
    K_i = (np.exp(5.373 * (1 + Wi) * (1 - Tkr / T)) * Pkr / P) ** 1.0
    ml, TS_l_flag = 0, 0
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

    # Определение стабильности
    if ((TS_l_flag == 1 and TS_v_flag == 1) or
            (Sv <= 1 and TS_l_flag == 1) or
            (Sl <= 1 and TS_v_flag == 1) or
            (Sv < 1 and Sl <= 1)):
        if print_output:
            print('Стабильное состояние')
        Stable = 1
        TestPTF = 1
    else:
        if print_output:
            print('Нестабильное состояние')
        Stable = 0
        TestPTF = 0

    # Flash расчет
    W = 0.5
    if Stable == 0 or TestPTF == 1:
        Kst_v = np.sum((K_iv - 1) ** 2)
        Kst_l = np.sum((K_il - 1) ** 2)
        K_i = K_il if Kst_l > Kst_v else K_iv

        m, eps_f = 0, 1
        MaxIterationFlash = 500
        W_old, Rr_old = 0, 0

        while eps_f > 1e-6 and m < MaxIterationFlash:
            W = findroot(z, K_i)
            x_i = z / (1 + W * (K_i - 1))
            y_i = K_i * x_i
            fw_i, Z_v = calculate_z(R, Tkr, Pkr, si, Wi, T, P, y_i, N, 1)
            fl_i, Z_l = calculate_z(R, Tkr, Pkr, si, Wi, T, P, x_i, N, 0)

            df_lv = np.zeros(N)
            Rr = fl_i / fw_i

            if m <= N:
                for t in range(N):
                    if fl_i[t] != 0:
                        K_i[t] *= (fl_i[t] / fw_i[t]) ** 1
                        df_lv[t] = (fl_i[t] / fw_i[t]) - 1

            if m > 1:
                Crit3 = np.sum((Rr - 1) ** 2)
                Crit1 = Crit3 / np.sum((Rr_old - 1) ** 2)
                Crit2 = abs(W - W_old)

                if m > N and Crit1 > 0.8 and Crit2 < 0.1 and Crit3 < 0.001:
                    for t in range(N):
                        if fl_i[t] != 0:
                            K_i[t] *= (fl_i[t] / fw_i[t]) ** 6.0
                            df_lv[t] = (fl_i[t] / fw_i[t]) - 1
                elif m > N:
                    for t in range(N):
                        if fl_i[t] != 0:
                            K_i[t] *= (fl_i[t] / fw_i[t]) ** 1
                            df_lv[t] = (fl_i[t] / fw_i[t]) - 1

            W_old, Rr_old = W, Rr
            eps_f = np.max(np.abs(df_lv))
            m += 1

            Stable = 0
            if m > 5 and abs(W) > 2:
                m = 400
                Stable = 1

            if m > 80 and abs(W - 0.5) > 0.501:
                m = 400
                Stable = 1

            if Stable == 1 or abs(W - 0.5) > 0.5:
                Volume = 1000 * (Z_init * R * T / P)
                if T_pkr < 260:
                    W = 1
                    x_i = np.zeros(N)
                    y_i = z
                    Z_v = Z_init
                    Z_l = 0
                else:
                    if Volume * T ** 2 > Control_phase:
                        W = 1
                        x_i = np.zeros(N)
                        y_i = z
                        Z_v = Z_init
                        Z_l = 0
                    else:
                        W = 0
                        x_i = z
                        y_i = np.zeros(N)
                        Z_v = 0
                        Z_l = Z_init

    if print_output:
        print("\nРезультаты расчета:")
        print("W =", W)
        print("Zv =", Z_v)
        print("Zl =", Z_l)
        print("xi =", x_i)
        print("yi =", y_i)
        print(f"Время выполнения: {time.time() - start_time:.4f} секунд")

    return W, abs(W - 0.5) > 0.501


def build_phase_diagram(user_values, T_range, P_range):
    """Построение фазовой диаграммы с пропуском точки T=340, P=14"""
    start_time = time.time()
    stability_map = np.zeros((len(P_range), len(T_range)))

    for i, P in enumerate(P_range):
        for j, T in enumerate(T_range):
            # Пропускаем точку T=340, P=14 и заменяем на P=13.9
            if np.isclose(T, 340) and np.isclose(P, 14):
                _, is_stable = calculate_single_point(340, 13.9, user_values, print_output=False)
            else:
                _, is_stable = calculate_single_point(T, P, user_values, print_output=False)
            stability_map[i, j] = 1 if is_stable else 0

    # Визуализация
    plt.figure(figsize=(10, 6))
    plt.contourf(T_range, P_range, stability_map, levels=[-0.5, 0.5, 1.5], colors=['red', 'blue'])
    plt.colorbar(label='Стабильность (1 - стабильно, 0 - нестабильно)')
    plt.xlabel('Температура (K)')
    plt.ylabel('Давление (MPa)')
    plt.title('Фазовая диаграмма (abs(W-0.5) > 0.501)\n(Точка T=340, P=14 заменена на P=13.9)')
    plt.grid(True)
    plt.show()

    print(f"Общее время построения диаграммы: {time.time() - start_time:.2f} секунд")


# Параметры системы
user_values = {
    'C1': 0.70,
    'C3': 0.15,
    'nC5': 0.15
}

# Пример расчета для одной точки
print("Расчет для одной точки:")
T, P = 343.15, 13.806
calculate_single_point(T, P, user_values)

# Построение фазовой диаграммы
print("\nПостроение фазовой диаграммы:")
T_min, T_max = 200, 400  # K
P_min, P_max = 1, 20  # MPa
T_step = 10  # Шаг по температуре
P_step = 1  # Шаг по давлению

T_range = np.arange(T_min, T_max + T_step, T_step)
P_range = np.arange(P_min, P_max + P_step, P_step)

build_phase_diagram(user_values, T_range, P_range)