import os
import cv2
import numpy as np
import pandas as pd
from scipy.stats import qmc
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
from scipy.optimize import root, curve_fit
from ddeint import ddeint
from scipy.interpolate import interp1d
from matplotlib.backends.backend_pdf import PdfPages

# 从文件中读取参数
def load_params_from_file(file_path):
    params_df = pd.read_excel(file_path, header=0)  # 从第二行开始读取数据，第一行作为标题
    return params_df

def model_ode(y, t, IPTG, ATC, k_IPTG, k_ATC, k_LT, k_TL, b_T, b_L, a_T, a_L, n_E):
    E, T, L = y

    f_SALI = SALI  # 假设SALI的值（之前代码里没有给出定义）
    f_IPTG = k_IPTG**2 / (IPTG**2 + k_IPTG**2)
    f_ATC = k_ATC**2 / (ATC**2 + k_ATC**2)

    dEdt = b_E + (a_E * f_SALI**n_NE) / (f_SALI**n_NE + k_NE**n_NE) - delta * E
    dTdt = b_T + a_T * (k_LT**n_LT / ((f_IPTG * L)**n_LT + k_LT**n_LT)) * (E**n_E / (E**n_E + k_E**n_E)) - delta * T
    dLdt = b_L + a_L * (k_TL**n_TL / ((f_ATC * T)**n_TL + k_TL**n_TL)) * (E**n_E / (E**n_E + k_E**n_E)) - delta * L

    return [dEdt, dTdt, dLdt]

def simulate_trajectory_ode(time_points, IPTG, ATC, initial_E, initial_T, initial_L, params):
    k_IPTG, k_ATC, k_LT, k_TL, b_T, b_L, a_T, a_L, n_E = params
    solution = odeint(model_ode, [initial_E, initial_T, initial_L], time_points, args=(IPTG, ATC, k_IPTG, k_ATC, k_LT, k_TL, b_T, b_L, a_T, a_L, n_E))
    return solution[:, 0], solution[:, 1], solution[:, 2]

def visualize_simulation_to_pdf(file_path, num_iterations=2):
    # 读取参数文件
    params_df = load_params_from_file(file_path)

    # 创建PDF文件保存图像
    pdf_filename = os.path.join(os.path.dirname(file_path), 'simulation_results.pdf')
    with PdfPages(pdf_filename) as pdf:
        # Excel记录T_flag和L_flag
        result_data = []

        # 时间点
        time_points = np.linspace(0, 4000, 4000)

        # T0 和 L0 扫描范围
        T0_values = np.linspace(0, 6000, 41)  # 20个点，范围从0到3001
        L0_values = np.linspace(0, 6000, 41)  # 20个点，范围从0到3001
        T0_grid, L0_grid = np.meshgrid(T0_values, L0_values)

        # 进行1000次循环
        for iteration in range(num_iterations):
            params = params_df.iloc[iteration].values[:-1]  # 获取每行的前9个参数值，忽略最后一列best_error

            # 获取k_IPTG的值
            k_IPTG_value = params[0]  # 假设k_IPTG是第一个参数

            # 初始化图形
            fig, axes = plt.subplots(4, 4, figsize=(20, 20))  # 4x4网格，16张子图
            axes = axes.flatten()  # 使axes成为一维数组

            fig.suptitle(f'k_IPTG={k_IPTG_value}', fontsize=16)

            # 对每一组 (IPTG, ATC) 条件，模拟并绘制结果
            for i, (IPTG, ATC) in enumerate(IPTG_ATC_conditions):
                T_flag = 0  # 每个条件独立统计
                L_flag = 0  # 每个条件独立统计

                # 对每一个T0和L0组合进行模拟
                end_states_T = []
                end_states_L = []
                for sim_idx in range(T0_grid.size):
                    E0 = 0.1
                    T0 = T0_grid.flat[sim_idx]  # 从网格中取T0
                    L0 = L0_grid.flat[sim_idx]  # 从网格中取L0

                    # 模拟ODE轨迹
                    E_sim, T_sim, L_sim = simulate_trajectory_ode(time_points, IPTG, ATC, E0, T0, L0, params)

                    # 记录末态的T和L值
                    end_states_T.append(T_sim[-1])
                    end_states_L.append(L_sim[-1])

                    # 根据末态的T和L值判断T_flag和L_flag
                    if T_sim[-1] > L_sim[-1]:
                        T_flag += 1
                    else:
                        L_flag += 1

                # 绘制每一组IPTG, ATC条件的图
                for j, (T_val, L_val) in enumerate(zip(end_states_T, end_states_L)):
                    if T_val > L_val:
                        axes[i].scatter(L_val, T_val, color='green', s=20, alpha=0.5)
                    else:
                        axes[i].scatter(L_val, T_val, color='red', s=20, alpha=0.5)

                axes[i].set_title(f'IPTG={IPTG}, ATC={ATC}')
                axes[i].set_xlabel('Final L Expression (a.u.)')
                axes[i].set_ylabel('Final T Expression (a.u.)')
                axes[i].set_xlim(0, 6000)
                axes[i].set_ylim(0, 6000)
                axes[i].grid(True)

                # 记录当前条件的T_flag和L_flag
                result_data.append([iteration + 1, IPTG, ATC, T_flag, L_flag])

            # 保存当前循环的16张图到PDF
            pdf.savefig(fig)
            plt.close(fig)

        # 将T_flag和L_flag值保存到Excel文件
        result_df = pd.DataFrame(result_data, columns=['Iteration', 'IPTG', 'ATC', 'T_flag', 'L_flag'])
        result_df.to_excel(os.path.join(os.path.dirname(file_path), 'T_L_flags.xlsx'), index=False)


# 文件路径
file_path = r'C:\bistable_processing\best_params_1000_iterations_1110.xlsx'

IPTG_ATC_conditions = [
    (100.0, 0.0), (100.0, 3.125), (100.0, 6.25), (100.0, 12.5),
    (50.0, 0.0), (50.0, 3.125), (50.0, 6.25), (50.0, 12.5),
    (25.0, 0.0), (25.0, 3.125), (25.0, 6.25), (25.0, 12.5),
    (0.0, 0.0), (0.0, 3.125), (0.0, 6.25), (0.0, 12.5),
]
SALI = 300.0

b_E, a_E, n_NE, k_NE = 1.71, 61.98, 0.73, 18.0
k_E, delta = 824.0, 0.013
n_IPTG, n_ATC, n_LT, n_TL = 2.0, 2.0, 2.0, 2.0

# 运行模拟并保存结果
visualize_simulation_to_pdf(file_path, num_iterations=2)