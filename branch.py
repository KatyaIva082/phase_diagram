import numpy as np
import matplotlib.pyplot as plt
from Constants import critical_constants
from calculate import calculate_z
from findroot import findroot
import time

def calculate_stability(T, P, user_values):
    """Расчет стабильности для одной точки (T,P)."""
    R = 0.00831675
    # Получаем параметры для каждого компонента
    N = len(user_values)
    z = np.array(list(user_values.values()))
    z = z / np.sum(z)
    Tkr, Pkr, Wi, si, Vkr = [], [], [], [], []
    for component in user_values:
        if component in critical_constants:
            crit = critical_constants[component]
            Tkr.append(crit['Tkr'])
            Pkr.append(crit['Pkr'])
            Wi.append(crit['Wi'])
            si.append(crit['si'])
            Vkr.append(crit['Vkr'])
        else:
            Tkr.append(0)
            Pkr.append(0)
            Wi.append(0)
            si.append(0)
            Vkr.append(0)
    Tkr, Pkr, Wi, si, Vkr = map(np.array, [Tkr, Pkr, Wi, si, Vkr])
    # Расчет стабильности
    K_i = (np.exp(5.373 * (1 + Wi) * (1 - Tkr / T)) * Pkr / P) ** 1.0
    fz_i, Z_init = calculate_z(R, Tkr, Pkr, si, Wi, T, P, z, N, 1)
    # Проверка газовой фазы
    m, TS_v_flag = 0, 0
    while m < 30:
        Yi_v = z * K_i
        Sv = np.sum(Yi_v)
        y_i = Yi_v / Sv
        fw_i, Z_v = calculate_z(R, Tkr, Pkr, si, Wi, T, P, y_i, N, 1)
        Ri = fz_i / (Sv * fw_i)
        Ri_v = np.sum((Ri - 1) ** 2)
        if Ri_v < 1e-12:
            m = 30
        K_i = K_i * Ri
        TS_v = np.sum(np.log(K_i) ** 2)
        if TS_v < 1e-4:
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
        return True, K_iv, K_il, Z_init
    else:
        return False, K_iv, K_il, Z_init

def calculate_flash(T, P, user_values, K_iv, K_il, Z_init):
    """Flash расчет для одной точки (T,P)."""
    R = 0.00831675
    # Получаем параметры для каждого компонента
    N = len(user_values)
    z = np.array(list(user_values.values()))
    z = z / np.sum(z)
    Tkr, Pkr, Wi, si, Vkr = [], [], [], [], []
    for component in user_values:
        if component in critical_constants:
            crit = critical_constants[component]
            Tkr.append(crit['Tkr'])
            Pkr.append(crit['Pkr'])
            Wi.append(crit['Wi'])
            si.append(crit['si'])
            Vkr.append(crit['Vkr'])
        else:
            Tkr.append(0)
            Pkr.append(0)
            Wi.append(0)
            si.append(0)
            Vkr.append(0)
    Tkr, Pkr, Wi, si, Vkr = map(np.array, [Tkr, Pkr, Wi, si, Vkr])
    V_pkr = np.dot(z, Vkr)
    T_pkr = np.dot(z, Tkr)
    Control_phase = V_pkr * T_pkr * T_pkr
    Kst_v = np.sum((K_iv - 1) ** 2)
    Kst_l = np.sum((K_il - 1) ** 2)
    K_i = K_il if Kst_l > Kst_v else K_iv
    m, eps_f = 0, 1
    MaxIterationFlash = 500
    W_old, Rr_old = 0, 0
    while eps_f > 1e-5 and m < MaxIterationFlash:
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
    # Сохраняем "чистое" значение W в Wsat
    Wsat = W
    return W, Wsat

def main():
    """Основная функция расчета с двумя ветками."""
    user_values = {
        'C1': 0.7,
        'C3': 0.2,
        'nC5': 0.1,
    }
    P_range = np.arange(30, 0, -0.005)
    T_range = np.arange(230, 370, 1)
    total_start = time.time()

    # Отдельные списки для верхней и нижней ветки
    T_list_upper, Pstable_list_upper, Psat_list_upper = [], [], []
    T_list_lower, Pstable_list_lower, Psat_list_lower = [], [], []

    for T in T_range:
        print(f"Обработка температуры: {T} K")
        Pstable_high = None
        Pstable_low = None
        found_first = False

        # 1. Поиск двух границ стабильности
        for P in P_range:
            is_stable, K_iv, K_il, Z_init = calculate_stability(T, P, user_values)
            if not found_first and not is_stable:
                Pstable_high = P
                found_first = True
            elif found_first and is_stable:
                Pstable_low = P
                break

        # 2. Flash расчет вверх от верхней границы
        Psat_high = None
        if Pstable_high:
            flash_P_range_high = np.arange(Pstable_high, 30.005, 0.005)
            for P_flash in flash_P_range_high:
                W, Wsat = calculate_flash(T, P_flash, user_values, K_iv, K_il, Z_init)
                if 0 < W < 1:
                    Psat_high = P_flash
                else:
                    break
            if Psat_high:
                T_list_upper.append(T)
                Pstable_list_upper.append(Pstable_high)
                Psat_list_upper.append(Psat_high)

        # 3. Flash расчет вниз от нижней границы
        Psat_low = None
        if Pstable_low:
            flash_P_range_low = np.arange(Pstable_low, 0, -0.005)
            for P_flash in flash_P_range_low:
                W, Wsat = calculate_flash(T, P_flash, user_values, K_iv, K_il, Z_init)
                if 0 < W < 1:
                    Psat_low = P_flash
                else:
                    break
            if Psat_low:
                T_list_lower.append(T)
                Pstable_list_lower.append(Pstable_low)
                Psat_list_lower.append(Psat_low)

    # Построение графика с двумя ветками
    plt.figure(figsize=(12, 6))

    # Левая панель: фазовые кривые
    plt.subplot(1, 2, 1)
    if T_list_upper:
        plt.plot(T_list_upper, Pstable_list_upper, label="Pstable (верх)", marker='o', linestyle='-', color='blue')
        plt.plot(T_list_upper, Psat_list_upper, label="Psat (верх)", marker='x', linestyle='--', color='red')
    if T_list_lower:
        plt.plot(T_list_lower, Pstable_list_lower, label="Pstable (низ)", marker='o', linestyle='-', color='cyan')
        plt.plot(T_list_lower, Psat_list_lower, label="Psat (низ)", marker='x', linestyle='--', color='orange')
    plt.xlabel("Температура (K)")
    plt.ylabel("Давление (MPa)")
    plt.title("Фазовая диаграмма: две ветки")
    plt.legend()
    plt.grid(True)

    # Правая панель: разница давлений
    plt.subplot(1, 2, 2)
    if T_list_upper:
        delta_P_upper = [ps - pst for ps, pst in zip(Psat_list_upper, Pstable_list_upper)]
        plt.plot(T_list_upper, delta_P_upper, label="ΔP верх", marker='o', linestyle='-', color='green')
    if T_list_lower:
        delta_P_lower = [ps - pst for ps, pst in zip(Psat_list_lower, Pstable_list_lower)]
        plt.plot(T_list_lower, delta_P_lower, label="ΔP низ", marker='x', linestyle='--', color='purple')
    plt.xlabel("Температура (K)")
    plt.ylabel("ΔP (MPa)")
    plt.title("Разность Psat - Pstable")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    total_time = time.time() - total_start
    print(f"\nОбщее время выполнения: {total_time:.2f} секунд")