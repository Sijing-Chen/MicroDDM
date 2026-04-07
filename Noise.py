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

plt.rcParams['figure.dpi'] = 300
# 字体调整
plt.rcParams['font.family'] = 'sans-serif'
#plt.rcParams['font.sans-serif'] = ['Arial']  # 如果要显示中文字体,则在此处设为：simhei,Arial Unicode MS
#plt.rcParams['font.weight'] = 'light'
plt.rcParams['axes.unicode_minus'] = False  # 坐标轴负号显示
plt.rcParams['axes.titlesize'] = 12  # 标题字体大小
plt.rcParams['axes.labelsize'] = 10  # 坐标轴标签字体大小
plt.rcParams['xtick.labelsize'] = 8  # x轴刻度字体大小
plt.rcParams['ytick.labelsize'] = 8  # y轴刻度字体大小
plt.rcParams['legend.fontsize'] = 8
# 线条调整
#plt.rcParams['axes.linewidth'] = 0.8#100条时linewidth1
# 刻度在内，设置刻度字体大小
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
# 设置输出格式为PDF
#plt.rcParams['savefig.format'] = 'pdf'
plt.rcParams['figure.autolayout'] = True
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


# 读取数据
def load_from_excel(filename):
    all_results = []
    excel_file = pd.ExcelFile(filename)

    for sheet_name in excel_file.sheet_names:
        group_df = pd.read_excel(filename, sheet_name=sheet_name)
        group_id = int(sheet_name.split('_')[1])

        data = {}
        for col in group_df.columns:
            if '_time' in col:
                channel = col.replace('_time', '')
                data[channel] = {
                    'time_points': group_df[f'{channel}_time'].dropna().values,
                    'mean_values': group_df[f'{channel}_mean'].dropna().values,
                    'std_values': group_df[f'{channel}_std'].dropna().values
                }

        all_results.append({'group_id': group_id, 'data': data})

    return all_results

all_group_results_630 = load_from_excel('E:/250630/all_group_results_630.xlsx')
red_background = 542.4033861
green_background = 569.7855446
redmax = 4690.134248
greenmax = 3552.732481


def f_E(t, m_E, d_E):
    return 1 + ((delta + m_E) * np.exp(-(delta + d_E) * t) - (delta + d_E) * np.exp(-(delta + m_E) * t)) / (
                d_E - m_E + 1e-9)


def f_T(t, m_T, d_T):
    return 1 + ((delta + m_T) * np.exp(-(delta + d_T) * t) - (delta + d_T) * np.exp(-(delta + m_T) * t)) / (
                d_T - m_T + 1e-9)


def f_L(t, m_L, d_L):
    return 1 + ((delta + m_L) * np.exp(-(delta + d_L) * t) - (delta + d_L) * np.exp(-(delta + m_L) * t)) / (
                d_L - m_L + 1e-9)


def model(it, y, f_SALI, f_IPTG, f_ATC, idelay, E_init, T_init, L_init, dt):
    t = it * dt
    E, T, L = y[it, :]
    E_tau_T = y[it - idelay[0], 0] if it > idelay[0] else E_init
    E_tau_L = y[it - idelay[1], 0] if it > idelay[1] else E_init
    T_tau_L = y[it - idelay[1], 1] if it > idelay[1] else T_init
    L_tau_T = y[it - idelay[0], 2] if it > idelay[0] else L_init

    dE_dt = b_E + f_E(t, m_E, d_E) * (a_E * f_SALI ** n_NE) / (f_SALI ** n_NE + k_NE ** n_NE) - delta * E
    dT_dt = b_T + f_T(t, m_T, d_T) * a_T * (k_LT ** n_LT / ((f_IPTG * L_tau_T) ** n_LT + k_LT ** n_LT)) * (
            E_tau_T ** n_E / (E_tau_T ** n_E + k_E ** n_E)) - delta * T
    dL_dt = b_L + f_L(t, m_L, d_L) * a_L * (k_TL ** n_TL / ((f_ATC * T_tau_L) ** n_TL + k_TL ** n_TL)) * (
            E_tau_L ** n_E / (E_tau_L ** n_E + k_E ** n_E)) - delta * L
    return [dE_dt, dT_dt, dL_dt]


def simulate_trajectory(E0, T0, L0, time_points, IPTG, ATC, noise_std_init, noise_std_dyn):
    dt = 1.0
    max_time = time_points[-1]
    itend = int(max_time / dt) + 1

    f_SALI = SALI
    f_IPTG = k_IPTG ** n_IPTG / (IPTG ** n_IPTG + k_IPTG ** n_IPTG)
    f_ATC = k_ATC ** n_ATC / (ATC ** n_ATC + k_ATC ** n_ATC)
    tau_T = 1 / (delta + m_T) + 1 / (delta + d_T)
    tau_L = 1 / (delta + m_L) + 1 / (delta + d_L)
    idelay = np.array([int(tau_T / dt), int(tau_L / dt)], dtype=np.int32)

    init_noise = np.clip(np.random.normal(0, 1, 3), -2.0, 2.0)
    E_init = max(0, (E0 + noise_std_init[0] * init_noise[0]))
    T_init = max(0, (T0 + noise_std_init[1] * init_noise[1]))
    L_init = max(0, (L0 + noise_std_init[2] * init_noise[2]))
    noise_matrix_dyn = np.random.normal(0, 1, (3, itend))
    noise_matrix_dyn = np.clip(noise_matrix_dyn, -2.0, 2.0)

    y = np.zeros(shape=(itend, 3))
    y[0, :] = [E_init, T_init, L_init]

    for it in range(0, itend - 1):
        dydt = model(it, y, f_SALI, f_IPTG, f_ATC, idelay, E_init, T_init, L_init, dt)
        y[it + 1, :] = y[it, :] + dt * np.array(dydt)
        y[it + 1, 0] = y[it + 1, 0] + noise_std_dyn[0] * np.sqrt(dt) * noise_matrix_dyn[0, it]
        y[it + 1, 1] = y[it + 1, 1] + noise_std_dyn[1] * np.sqrt(dt) * noise_matrix_dyn[1, it]
        y[it + 1, 2] = y[it + 1, 2] + noise_std_dyn[2] * np.sqrt(dt) * noise_matrix_dyn[2, it]
        y[it + 1, :] = np.maximum(0, y[it + 1, :])

    step = int(round((time_points[1] - time_points[0]) / dt))
    E_sim = y[:itend:step, 0]
    T_sim = y[:itend:step, 1]
    L_sim = y[:itend:step, 2]

    return E_sim, T_sim, L_sim


def visualize_simulation(experimental_data, IPTG_ATC_conditions, noise_std_init, noise_std_dyn, E0=0.1):
    for i, (group, (IPTG, ATC)) in enumerate(zip(experimental_data, IPTG_ATC_conditions)):
        group_data = group['data']
        exp_time = next(iter(group_data.values()))['time_points']
        time_points = np.arange(0, exp_time[-1] + 1, 1)
        num_sims = 1000

        has_set1 = 'red_in_red' in group_data and 'green_in_red' in group_data
        has_set2 = 'red_in_green' in group_data and 'green_in_green' in group_data

        tasks = []
        if has_set1 and has_set2:
            tasks.append(('set1', int(num_sims / 2)))
            tasks.append(('set2', num_sims - int(num_sims / 2)))
        elif has_set1:
            tasks.append(('set1', num_sims))
        elif has_set2:
            tasks.append(('set2', num_sims))

        left_T, left_L = [], []
        right_T, right_L = [], []

        for set_type, n in tasks:
            if set_type == 'set1':
                T0 = group_data['green_in_red']['mean_values'][0]
                L0 = group_data['red_in_red']['mean_values'][0]
            else:
                T0 = group_data['green_in_green']['mean_values'][0]
                L0 = group_data['red_in_green']['mean_values'][0]

            for _ in range(n):
                _, T_sim, L_sim = simulate_trajectory(E0, T0, L0, time_points, IPTG, ATC, noise_std_init, noise_std_dyn)
                if T_sim[-1] < L_sim[-1]:
                    left_T.append(T_sim)
                    left_L.append(L_sim)
                else:
                    right_T.append(T_sim)
                    right_L.append(L_sim)

        ncol = 2 if (left_T and right_T) else 1
        fig, axes = plt.subplots(1, ncol, figsize=(2.2, 1.65), sharey=True, squeeze=False, gridspec_kw={'wspace': 0})

        if left_T:
            ax = axes[0, 0]
            T_mean, T_std = np.mean(left_T, axis=0) / greenmax, np.std(left_T, axis=0) / greenmax
            L_mean, L_std = np.mean(left_L, axis=0) / redmax, np.std(left_L, axis=0) / redmax

            ax.plot(time_points, T_mean, color='turquoise', linewidth=1.0)
            ax.fill_between(time_points, np.maximum(0, T_mean - T_std), T_mean + T_std, color='turquoise', alpha=0.2,
                            linewidth=0)

            ax.plot(time_points, L_mean, color='red', linewidth=1.0)
            ax.fill_between(time_points, np.maximum(0, L_mean - L_std), L_mean + L_std, color='red', alpha=0.2,
                            linewidth=0)

            ax.text(0.05, 0.95, f'n={len(left_T)}', transform=ax.transAxes, fontsize=7, verticalalignment='top')

        if right_T:
            ax = axes[0, 1] if ncol == 2 else axes[0, 0]
            T_mean, T_std = np.mean(right_T, axis=0) / greenmax, np.std(right_T, axis=0) / greenmax
            L_mean, L_std = np.mean(right_L, axis=0) / redmax, np.std(right_L, axis=0) / redmax

            ax.plot(time_points, T_mean, color='darkgreen', linewidth=1.0)
            ax.fill_between(time_points, np.maximum(0, T_mean - T_std), T_mean + T_std, color='darkgreen', alpha=0.2,
                            linewidth=0)

            ax.plot(time_points, L_mean, color='lightsalmon', linewidth=1.0)
            ax.fill_between(time_points, np.maximum(0, L_mean - L_std), L_mean + L_std, color='lightsalmon', alpha=0.2,
                            linewidth=0)

            ax.text(0.05, 0.95, f'n={len(right_T)}', transform=ax.transAxes, fontsize=7, verticalalignment='top')

        for j, ax in enumerate(axes.flatten()):
            ax.set_xlim(0, 1100)
            ax.set_ylim(0, 1.5)
            ax.tick_params(axis='both', which='major', labelsize=7)
            ax.locator_params(axis='x', nbins=5)
            ax.locator_params(axis='y', nbins=5)

            if j == ncol - 1:
                ax.yaxis.tick_right()
                ax.yaxis.set_label_position("right")
                ax.tick_params(axis='y', which='both', right=True, labelright=True, left=False, labelleft=False)
            else:
                ax.tick_params(axis='y', which='both', left=False, labelleft=False)

        plt.tight_layout()
        plt.savefig(fr'E:\250630\result\NoiseSim_IPTG={IPTG}_ATC={ATC}.pdf', format='pdf', bbox_inches='tight')
        plt.show()
        plt.close()


# 实验条件
IPTG_ATC_conditions = [
    (0.0, 0.0), (0.0, 3.125), (0.0, 6.25), (0.0, 12.5),
    (25.0, 0.0), (25.0, 3.125), (25.0, 6.25), (25.0, 12.5),
    (50.0, 0.0), (50.0, 3.125), (50.0, 6.25), (50.0, 12.5),
    (100.0, 0.0), (100.0, 3.125), (100.0, 6.25), (100.0, 12.5)
]
SALI = 300.0

# 稳态参数
df = pd.read_excel(r"E:\250630\result\bistable steady params 260321\bsp166.xlsx")
for i, row in df.iterrows():
    globals()[row['Parameter']] = row['Value']

b_E, a_E, n_NE, k_NE = 1.71, 61.98, 0.73, 18.0
k_E, delta = 824.0, 0.013
n_IPTG, n_ATC, n_LT, n_TL = 2.0, 2.0, 2.0, 2.0
b_T, a_T, k_LT, k_IPTG = b_T, a_T, k_LT, k_IPTG
b_L, a_L, k_TL, k_ATC = b_L, a_L, k_TL, k_ATC

# 动态参数
df = pd.read_excel(r"E:\250630\result\best_dynamic_params_bsp166_top20.xlsx")
for i, row in df.iterrows():
    globals()[row['Parameter']] = row['Value']

m_E, m_T, m_L = m_E, m_T, m_L
d_E, d_T, d_L = d_E, d_T, d_L
n_E = n_E

# 噪声参数
noise_std_init = [10.0, 100.0, 100.0]
noise_std_dyn = [2.0, 4.0, 4.0]


visualize_simulation(all_group_results_630, IPTG_ATC_conditions, noise_std_init, noise_std_dyn)


