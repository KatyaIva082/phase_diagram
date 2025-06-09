import numpy as np
from Constants import critical_constants
from calculate import calculate_z
import time
import matplotlib.pyplot as plt


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
        return True  # Система стабильна
    else:
        return False  # Система нестабильна


def find_stability_boundary(T_range, P_range, user_values):
    """Нахождение границы между стабильной и нестабильной областями."""
    stability_boundary_forward = []  # Для хранения точек границы при движении вперед
    stability_boundary_backward = []  # Для хранения точек границы при движении назад

    # Флаг для направления движения по температуре (True = вперед, False = назад)
    forward = True

    # Цикл по температуре
    T_index = 0
    while T_index < len(T_range) and T_index >= 0:
        T = T_range[T_index]

        # Определяем направление движения по давлению
        if forward:
            pressure_direction = -1  # Давление уменьшается (сверху вниз)
        else:
            pressure_direction = 1  # Давление увеличивается (снизу вверх)

        # Цикл по давлению
        previous_stable = None  # Предыдущее состояние стабильности
        for P in P_range[::pressure_direction]:  # Итерация по давлению в зависимости от направления
            is_stable = calculate_stability(T, P, user_values)

            # Если произошел переход от стабильного к нестабильному или наоборот
            if previous_stable is not None and is_stable != previous_stable:
                # Сохраняем точку перехода
                if forward:
                    stability_boundary_forward.append((T, P))
                else:
                    stability_boundary_backward.append((T, P))

                # Выполняем обратный шаг по давлению для точной трассировки границы
                P_back = P + 0.1 * pressure_direction  # Шаг назад
                while calculate_stability(T, P_back, user_values):
                    P_back -= 0.1 * pressure_direction
                if forward:
                    stability_boundary_forward.append((T, P_back))
                else:
                    stability_boundary_backward.append((T, P_back))

                break  # Переходим к следующей температуре

            previous_stable = is_stable

        # Обновляем индекс температуры
        if forward:
            T_index += 1
            if T_index >= len(T_range):  # Достигли максимума, меняем направление
                forward = False
                T_index -= 2  # Начинаем движение назад
        else:
            T_index -= 1
            if T_index < 0:  # Достигли минимума, завершаем цикл
                break

    return stability_boundary_forward, stability_boundary_backward


def plot_boundary(boundary_forward, boundary_backward):
    """Построение графика границы стабильности."""
    # Точки для движения вперед
    T_values_forward = [point[0] for point in boundary_forward]
    P_values_forward = [point[1] for point in boundary_forward]

    # Точки для движения назад
    T_values_backward = [point[0] for point in boundary_backward]
    P_values_backward = [point[1] for point in boundary_backward]

    plt.figure(figsize=(8, 6))
    plt.scatter(T_values_forward, P_values_forward, color='blue', label='Граница стабильности')
    plt.scatter(T_values_backward, P_values_backward, color='blue')  # Те же цвета
    plt.xlabel("Температура (K)")
    plt.ylabel("Давление (MPa)")
    plt.title("Граница стабильности")
    plt.grid(True)
    plt.legend()
    plt.show()


def main():
    """Основная функция расчета."""
    # Параметры системы
    user_values = {
        'C1': 0.70,
        'C3': 0.15,
        'nC5': 0.15
    }

    # Диапазон расчетов
    T_range = np.arange(200.15, 400.15, 1)  # Температуры от 200 до 380 K с шагом 1
    P_range = np.arange(18, 1, -0.1)  # Давления от 18 до 1 MPa с шагом -0.1

    total_start = time.time()

    # Этап 1: Находим границу стабильности
    start_boundary_time = time.time()
    boundary_forward, boundary_backward = find_stability_boundary(T_range, P_range, user_values)
    end_boundary_time = time.time()

    print("\nГраница между стабильной и нестабильной областями:")
    print("T (K)\tP (MPa)")
    print("----------------")
    print("Движение вперед:")
    for T, P in boundary_forward:
        print(f"{T}\t{P}")
    print("\nДвижение назад:")
    for T, P in boundary_backward:
        print(f"{T}\t{P}")

    # Этап 2: Построение графика
    start_plot_time = time.time()
    plot_boundary(boundary_forward, boundary_backward)
    end_plot_time = time.time()

    # Вывод времени выполнения каждого этапа
    boundary_time = end_boundary_time - start_boundary_time
    plot_time = end_plot_time - start_plot_time
    total_time = time.time() - total_start

    print(f"\nВремя поиска границы стабильности: {boundary_time:.2f} секунд")
    print(f"Время построения графика: {plot_time:.2f} секунд")
    print(f"Общее время выполнения: {total_time:.2f} секунд")


if __name__ == "__main__":
    main()