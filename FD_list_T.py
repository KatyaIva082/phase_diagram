import numpy as np
import matplotlib.pyplot as plt
from Constants import critical_constants
from calculate import calculate_z
from findroot import findroot
import time


def calculate_stability(T, P, user_values):
    """Расчет стабильности для одной точки (T,P)."""
    R = 0.00831675

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
    K_iv = K_i.copy()

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
    K_il = K_i.copy()

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

    Wsat = W
    return W, Wsat


def run_flash_on_branch(T_range, P_range, user_values, flash_direction='up'):
    """
    Запуск флеш-расчёта по одной ветке.
    flash_direction = 'up' -> увеличиваем давление (идём вверх),
                      'down' -> уменьшаем давление (идём вниз)
    """
    results = []

    for T in T_range:
        print(f"Обработка температуры: {T} K")
        instability_pressure = None
        for i, P in enumerate(P_range):
            is_stable, K_iv, K_il, Z_init = calculate_stability(T, P, user_values)
            if not is_stable:
                instability_pressure = P
                break  # Первая найденная нестабильная точка

        if instability_pressure is None:
            continue  # Пропускаем, если не нашли нестабильность

        Pstable = round(instability_pressure, 4)

        # Определяем диапазон флеш-расчёта
        if flash_direction == 'up':
            flash_P_range = np.arange(Pstable, 30.005, 0.005)
        else:
            flash_P_range = np.arange(Pstable, 0.1, -0.005)

        Psat = None
        for P_flash in flash_P_range:
            W, Wsat = calculate_flash(T, P_flash, user_values, K_iv, K_il, Z_init)
            if 0 < W < 1:
                Psat = round(P_flash, 4)
            else:
                break  # Вышли из двухфазной области

        if Psat is not None:
            delta_P = round(Psat - Pstable, 4)
            results.append({
                "Temperature": round(T, 2),
                "Pstable": Pstable,
                "Psat": Psat,
                "Delta_P": delta_P
            })

    return results


def plot_combined(results_forward, results_backward):
    """Строит объединённый график с двумя ветками и разницей давлений."""

    # Собираем данные из результатов
    T_forward = [r['Temperature'] for r in results_forward]
    Pstable_forward = [r['Pstable'] for r in results_forward]
    Psat_forward = [r['Psat'] for r in results_forward]

    T_backward = [r['Temperature'] for r in results_backward]
    Pstable_backward = [r['Pstable'] for r in results_backward]
    Psat_backward = [r['Psat'] for r in results_backward]

    # График 1: Pstable и Psat по температуре
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(T_forward, Pstable_forward,
             label="Pstable (вперёд)", marker='o', linestyle='', color='green')
    plt.plot(T_forward, Psat_forward,
             label="Psat (вперёд)", marker='x', linestyle='', color='darkred')

    plt.plot(T_backward, Pstable_backward,
             label="Pstable (назад)", marker='o', linestyle='', color='green')
    plt.plot(T_backward, Psat_backward,
             label="Psat (назад)", marker='x', linestyle='', color='darkred')

    plt.xlabel("Температура (K)")
    plt.ylabel("Давление (MPa)")
    plt.title("Pstable (зеленые точки) и Psat (красные крестики)")
    plt.grid(True)
    plt.legend()

    # График 2: ΔP = Psat - Pstable
    plt.subplot(1, 2, 2)
    delta_P_forward = [r['Psat'] - r['Pstable'] for r in results_forward]
    delta_P_backward = [r['Psat'] - r['Pstable'] for r in results_backward]

    plt.plot(T_forward, delta_P_forward,
             label="ΔP (Ветка 1)", marker='x', linestyle='', color='red')
    plt.plot(T_backward, delta_P_backward,
             label="ΔP (Ветка 2)", marker='x', linestyle='', color='darkred')

    plt.axhline(y=0, color='black', linestyle='--', linewidth=0.8)

    plt.xlabel("Температура (K)")
    plt.ylabel("ΔP = Psat - Pstable (MPa)")
    plt.title("Разница между Psat и Pstable")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

def main():
    """Основная функция расчета."""
    user_values = {
        'N2': 0.00324,
        'CO2': 0.09148,
        'H2S': 0.01157,
        'C1': 0.771,
        'C2': 0.03226,
        'C3': 0.01444,
        'iC4': 0.00427,
        'nC4': 0.00825,
        'iC5': 0.00501,
        'nC5': 0.00427,
        'C6':  0.00766,
        'C7': 0.04655,
    }

    T_range = [
    328.74317615391,
    333.120282138698,
    335.799381992788,
    339.915184416677,
    344.080263628689,
    348.281679994817,
    352.502669368967,
    356.721684627257,
    360.911168406528,
    365.035962960718,
    369.051219987331,
    372.899604450101,
    376.507472057467,
    379.779501225161,
    382.590895780496,
    384.775562315553,
    386.245200836366,
    386.371869276056,
    385.543698878451,
    381.94329533669,
    371.188254582103,
    354.078065196193,
    338.806890235528,
    319.995666092991,
    315.464202251378,
    305.261249748842,
    300.00000050621,
    296.000001365768,
    292.000001059724,
    288.552457429618,
    288.000000337975,
    284.000000000001,
    280.000000000003,
    276.000000000003,
    272.0,
    270.48286475374,
    268.000000000004,
    266.336151123282,
    264.000000000071,
    260.000000000079,
    254.950702968088,
    236.613698579484,
    225.991884919576
]

    P_range_forward = np.arange(30, 0, -0.005)
    P_range_backward = np.arange(1, 30, 0.005)
    total_start = time.time()

    # --- Первая ветка: сверху вниз ---
    print("Запуск первой ветки: поиск нестабильности сверху вниз по давлению...")
    results_forward = run_flash_on_branch(T_range, P_range_forward, user_values, flash_direction='up')

    # --- Вторая ветка: снизу вверх ---
    print("Запуск второй ветки: поиск нестабильности снизу вверх по давлению...")
    results_backward = run_flash_on_branch(T_range, P_range_backward, user_values, flash_direction='down')

    # --- Вывод таблицы в консоль ---
    print("\nТаблица результатов:")
    print(f"{'Температура (K)':<15} {'Pstable (MPa)':<15} {'Psat (MPa)':<15} {'Delta_P (MPa)':<15}")
    print("-" * 60)
    for result in results_forward + results_backward:
        print(
            f"{result['Temperature']:<15} {result['Pstable']:<15.4f} {result['Psat']:<15.4f} {result['Delta_P']:<15.4f}"
        )

    # --- Построение графика ---
    print("Строим объединённый график...")
    plot_combined(results_forward, results_backward)

    total_time = time.time() - total_start
    print(f"\nОбщее время выполнения: {total_time:.2f} секунд")


if __name__ == "__main__":
    main()