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


#取主要细菌单稳态
del all_group_results_630[9]['data']['red_in_red']
del all_group_results_630[9]['data']['green_in_red']

del all_group_results_630[10]['data']['red_in_green']
del all_group_results_630[10]['data']['green_in_green']

del all_group_results_630[15]['data']['red_in_green']
del all_group_results_630[15]['data']['green_in_green']

#只拟合两个单稳态条件，此处是原代码基础上改动1/2
# all_group_results_630 = [all_group_results_630[3], all_group_results_630[12]]


def model_ode(y, t, IPTG, ATC):
    E, T, L = y

    f_SALI = SALI
    f_IPTG = k_IPTG ** n_IPTG / (IPTG ** n_IPTG + k_IPTG ** n_IPTG)
    f_ATC = k_ATC ** n_ATC / (ATC ** n_ATC + k_ATC ** n_ATC)

    dEdt = b_E + (a_E * f_SALI ** n_NE) / (f_SALI ** n_NE + k_NE ** n_NE) - delta * E
    dTdt = b_T + a_T * (k_LT ** n_LT / ((f_IPTG * L) ** n_LT + k_LT ** n_LT)) * (
                E ** n_E / (E ** n_E + k_E ** n_E)) - delta * T
    dLdt = b_L + a_L * (k_TL ** n_TL / ((f_ATC * T) ** n_TL + k_TL ** n_TL)) * (
                E ** n_E / (E ** n_E + k_E ** n_E)) - delta * L

    return [dEdt, dTdt, dLdt]


def simulate_trajectory_ode(time_points, IPTG, ATC, initial_E, initial_T, initial_L):
    solution = odeint(model_ode, [initial_E, initial_T, initial_L], time_points, args=(IPTG, ATC))
    return solution[:, 0], solution[:, 1], solution[:, 2]


def model_dde(Y, t, IPTG, ATC, E0, T0, L0):
    f_SALI = SALI
    f_IPTG = k_IPTG ** n_IPTG / (IPTG ** n_IPTG + k_IPTG ** n_IPTG)
    f_ATC = k_ATC ** n_ATC / (ATC ** n_ATC + k_ATC ** n_ATC)

    def f_E(t, m_E, d_E):
        return 1 + ((delta + m_E) * np.exp(-(delta + d_E) * t) - (delta + d_E) * np.exp(-(delta + m_E) * t)) / (
                    d_E - m_E)

    def f_T(t, m_T, d_T):
        return 1 + ((delta + m_T) * np.exp(-(delta + d_T) * t) - (delta + d_T) * np.exp(-(delta + m_T) * t)) / (
                    d_T - m_T)

    def f_L(t, m_L, d_L):
        return 1 + ((delta + m_L) * np.exp(-(delta + d_L) * t) - (delta + d_L) * np.exp(-(delta + m_L) * t)) / (
                    d_L - m_L)

    tau_T = 1 / (delta + m_T) + 1 / (delta + d_T)
    tau_L = 1 / (delta + m_L) + 1 / (delta + d_L)

    E, T, L = Y(t)
    E_tau_T = Y(t - tau_T)[0] if t > tau_T else E0
    E_tau_L = Y(t - tau_L)[0] if t > tau_L else E0
    T_tau_L = Y(t - tau_L)[1] if t > tau_L else T0
    L_tau_T = Y(t - tau_T)[2] if t > tau_T else L0

    dE_dt = b_E + f_E(t, m_E, d_E) * (a_E * f_SALI ** n_NE) / (f_SALI ** n_NE + k_NE ** n_NE) - delta * E
    dT_dt = b_T + f_T(t, m_T, d_T) * a_T * (k_LT ** n_LT / ((f_IPTG * L_tau_T) ** n_LT + k_LT ** n_LT)) * (
                E_tau_T ** n_E / (E_tau_T ** n_E + k_E ** n_E)) - delta * T
    dL_dt = b_L + f_L(t, m_L, d_L) * a_L * (k_TL ** n_TL / ((f_ATC * T_tau_L) ** n_TL + k_TL ** n_TL)) * (
                E_tau_L ** n_E / (E_tau_L ** n_E + k_E ** n_E)) - delta * L
    return [dE_dt, dT_dt, dL_dt]


def simulate_trajectory_dde(E0, T0, L0, time_points, IPTG, ATC):
    max_time = time_points[-1]
    dense_time_points = np.linspace(0, max_time, 1000)

    solution = ddeint(model_dde, lambda t: [E0, T0, L0], dense_time_points, fargs=(IPTG, ATC, E0, T0, L0))

    E_dense, T_dense, L_dense = solution[:, 0], solution[:, 1], solution[:, 2]
    E_interp = interp1d(dense_time_points, E_dense, kind='linear', bounds_error=False, fill_value="extrapolate")
    T_interp = interp1d(dense_time_points, T_dense, kind='linear', bounds_error=False, fill_value="extrapolate")
    L_interp = interp1d(dense_time_points, L_dense, kind='linear', bounds_error=False, fill_value="extrapolate")
    E_sim = E_interp(time_points)
    T_sim = T_interp(time_points)
    L_sim = L_interp(time_points)

    return E_sim, T_sim, L_sim


def visualize_all_fits(experimental_data, IPTG_ATC_conditions):
    for i, (group, (IPTG, ATC)) in enumerate(zip(experimental_data, IPTG_ATC_conditions)):
        group_data = group['data']
        time_points = next(iter(group_data.values()))['time_points']
        dense_time_points = np.linspace(0, time_points[-1], 1000)

        # plt.subplot(8, 2, i+1)
        plt.figure(figsize=(2.2, 1.65))
        # plt.title(f'IPTG={IPTG}, ATC={ATC}')#Condition {i+1}:

        if 'red_in_red' in group_data and 'green_in_red' in group_data:
            E0, T0, L0 = 0.1, group_data['green_in_red']['mean_values'][0] * greenmax, \
                              group_data['red_in_red']['mean_values'][0] * redmax
            E_ode, T_ode, L_ode = simulate_trajectory_ode(time_points, IPTG, ATC, E0, T0, L0)
            E_dde, T_dde, L_dde = simulate_trajectory_dde(E0, T0, L0, dense_time_points, IPTG, ATC)
            plt.errorbar(time_points, group_data['green_in_red']['mean_values'],
                         yerr=group_data['green_in_red']['std_values'], color='turquoise', fmt='o', alpha=0.7, ms=0.8,
                         capsize=1.0, capthick=0.5, elinewidth=0.5, label='T (set1) exp')
            plt.errorbar(time_points, group_data['red_in_red']['mean_values'],
                         yerr=group_data['red_in_red']['std_values'], color='red', fmt='o', alpha=0.7, ms=0.8,
                         capsize=1.0, capthick=0.5, elinewidth=0.5, label='L (set1) exp')
            plt.plot(time_points, T_ode / greenmax, color='turquoise', linestyle='--', label='T (set1) ode sim')
            plt.plot(time_points, L_ode / redmax, color='red', linestyle='--', label='L (set1) ode sim')
            plt.plot(dense_time_points, T_dde / greenmax, color='turquoise', linestyle='-', label='T (set1) dde sim')
            plt.plot(dense_time_points, L_dde / redmax, color='red', linestyle='-', label='L (set1) dde sim')

        if 'red_in_green' in group_data and 'green_in_green' in group_data:
            E0, T0, L0 = 0.1, group_data['green_in_green']['mean_values'][0] * greenmax, \
                              group_data['red_in_green']['mean_values'][0] * redmax
            E_ode, T_ode, L_ode = simulate_trajectory_ode(time_points, IPTG, ATC, E0, T0, L0)
            E_dde, T_dde, L_dde = simulate_trajectory_dde(E0, T0, L0, dense_time_points, IPTG, ATC)
            plt.errorbar(time_points, group_data['green_in_green']['mean_values'],
                         yerr=group_data['green_in_green']['std_values'], color='darkgreen', fmt='^', alpha=0.7, ms=0.8,
                         capsize=1.0, capthick=0.5, elinewidth=0.5, label='T (set2) exp')
            plt.errorbar(time_points, group_data['red_in_green']['mean_values'],
                         yerr=group_data['red_in_green']['std_values'], color='lightsalmon', fmt='^', alpha=0.7, ms=0.8,
                         capsize=1.0, capthick=0.5, elinewidth=0.5, label='L (set2) exp')
            plt.plot(time_points, T_ode / greenmax, color='darkgreen', linestyle='--', label='T (set2) ode sim')
            plt.plot(time_points, L_ode / redmax, color='lightsalmon', linestyle='--', label='L (set2) ode sim')
            plt.plot(dense_time_points, T_dde / greenmax, color='darkgreen', linestyle='-', label='T (set2) dde sim')
            plt.plot(dense_time_points, L_dde / redmax, color='lightsalmon', linestyle='-', label='L (set2) dde sim')

        plt.xlim(0, 1100)
        plt.ylim(0, 1.5)
        plt.locator_params(axis='x', nbins=5)
        plt.locator_params(axis='y', nbins=5)
        plt.gca().yaxis.tick_right()
        # plt.xlabel('Time(min)')
        # plt.ylabel('Expression(a.u.)')
        # plt.legend()
        plt.savefig(fr'E:\250630\result\NormalizedDDMbsp166Condition{i + 1}.pdf', format='pdf', bbox_inches='tight')
        plt.show()
        plt.close()

    # plt.tight_layout()
    # plt.show()


# 实验条件
IPTG_ATC_conditions = [
    (0.0, 0.0), (0.0, 3.125), (0.0, 6.25), (0.0, 12.5),
    (25.0, 0.0), (25.0, 3.125), (25.0, 6.25), (25.0, 12.5),
    (50.0, 0.0), (50.0, 3.125), (50.0, 6.25), (50.0, 12.5),
    (100.0, 0.0), (100.0, 3.125), (100.0, 6.25), (100.0, 12.5)
]
SALI = 300.0

# 稳态参数
# df = pd.read_excel(r"E:\250630\result\top_parameters_1110_global.xlsx")
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


visualize_all_fits(all_group_results_630, IPTG_ATC_conditions)
