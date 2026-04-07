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

#取主要细菌单稳态
del all_group_results_630[9]['data']['red_in_red']
del all_group_results_630[9]['data']['green_in_red']

del all_group_results_630[10]['data']['red_in_green']
del all_group_results_630[10]['data']['green_in_green']

del all_group_results_630[15]['data']['red_in_green']
del all_group_results_630[15]['data']['green_in_green']


#正向筛选
def f_E(t, m_E, d_E):
    return 1 + ((delta + m_E) * np.exp(-(delta + d_E) * t) - (delta + d_E) * np.exp(-(delta + m_E) * t)) / (
                d_E - m_E + 1e-9)

def f_T(t, m_T, d_T):
    return 1 + ((delta + m_T) * np.exp(-(delta + d_T) * t) - (delta + d_T) * np.exp(-(delta + m_T) * t)) / (
                d_T - m_T + 1e-9)

def f_L(t, m_L, d_L):
    return 1 + ((delta + m_L) * np.exp(-(delta + d_L) * t) - (delta + d_L) * np.exp(-(delta + m_L) * t)) / (
                d_L - m_L + 1e-9)

def model(it, y, f_SALI, f_IPTG, f_ATC, idelay, E0, T0, L0, dt, params):
    m_E, m_T, m_L, d_E, d_T, d_L, n_E = params

    t = it * dt
    E, T, L = y[it, :]
    E_tau_T = y[it - idelay[0], 0] if it > idelay[0] else E0
    E_tau_L = y[it - idelay[1], 0] if it > idelay[1] else E0
    T_tau_L = y[it - idelay[1], 1] if it > idelay[1] else T0
    L_tau_T = y[it - idelay[0], 2] if it > idelay[0] else L0

    dE_dt = b_E + f_E(t, m_E, d_E) * (a_E * f_SALI ** n_NE) / (f_SALI ** n_NE + k_NE ** n_NE) - delta * E
    dT_dt = b_T + f_T(t, m_T, d_T) * a_T * (k_LT ** n_LT / ((f_IPTG * L_tau_T) ** n_LT + k_LT ** n_LT)) * (
                E_tau_T ** n_E / (E_tau_T ** n_E + k_E ** n_E)) - delta * T
    dL_dt = b_L + f_L(t, m_L, d_L) * a_L * (k_TL ** n_TL / ((f_ATC * T_tau_L) ** n_TL + k_TL ** n_TL)) * (
                E_tau_L ** n_E / (E_tau_L ** n_E + k_E ** n_E)) - delta * L
    return [dE_dt, dT_dt, dL_dt]

def simulate_trajectory(E0, T0, L0, time_points, IPTG, ATC, params):
    dt = 1.0
    max_time = time_points[-1]
    itend = int(max_time / dt) + 1

    y = np.zeros(shape=(itend, 3))
    y[0, :] = [E0, T0, L0]

    m_E, m_T, m_L, d_E, d_T, d_L, n_E = params
    f_SALI = SALI
    f_IPTG = k_IPTG ** n_IPTG / (IPTG ** n_IPTG + k_IPTG ** n_IPTG)
    f_ATC = k_ATC ** n_ATC / (ATC ** n_ATC + k_ATC ** n_ATC)
    tau_T = 1 / (delta + m_T) + 1 / (delta + d_T)
    tau_L = 1 / (delta + m_L) + 1 / (delta + d_L)
    idelay = np.array([int(tau_T / dt), int(tau_L / dt)], dtype=np.int32)

    for it in range(0, itend - 1):
        dydt = model(it, y, f_SALI, f_IPTG, f_ATC, idelay, E0, T0, L0, dt, params)
        y[it + 1, :] = y[it, :] + dt * np.array(dydt)

    step = int(round((time_points[1] - time_points[0]) / dt))
    E_sim = y[:itend:step, 0]
    T_sim = y[:itend:step, 1]
    L_sim = y[:itend:step, 2]

    return E_sim, T_sim, L_sim

def objective_function(experimental_data, IPTG_ATC_conditions, params):
    total_error = 0

    for group, (IPTG, ATC) in zip(experimental_data, IPTG_ATC_conditions):
        group_data = group['data']
        time_points = next(iter(group_data.values()))['time_points']

        if 'red_in_red' in group_data and 'green_in_red' in group_data:
            E0 = 0.1
            T0 = group_data['green_in_red']['mean_values'][0]
            L0 = group_data['red_in_red']['mean_values'][0]
            E_sim, T_sim, L_sim = simulate_trajectory(E0, T0, L0, time_points, IPTG, ATC, params)
            T_exp = group_data['green_in_red']['mean_values']
            L_exp = group_data['red_in_red']['mean_values']

            error_T = np.sum((T_sim - T_exp) ** 2)  # / (group_data['green_in_red']['std_values']**2 + 1e-6))
            error_L = np.sum((L_sim - L_exp) ** 2)  # / (group_data['red_in_red']['std_values']**2 + 1e-6))
            total_error += error_T + error_L

        if 'red_in_green' in group_data and 'green_in_green' in group_data:
            E0 = 0.1
            T0 = group_data['green_in_green']['mean_values'][0]
            L0 = group_data['red_in_green']['mean_values'][0]
            E_sim, T_sim, L_sim = simulate_trajectory(E0, T0, L0, time_points, IPTG, ATC, params)
            T_exp = group_data['green_in_green']['mean_values']
            L_exp = group_data['red_in_green']['mean_values']

            error_T = np.sum((T_sim - T_exp) ** 2)  # / (group_data['green_in_green']['std_values']**2 + 1e-6))
            error_L = np.sum(
                (L_sim - L_exp) ** 2)  # / (group_data['red_in_green']['std_values']**2 + 1e-6))
            total_error += error_T + error_L

    return total_error

def lhsampling(n_dim,n_samples,bounds):
    sampler = qmc.LatinHypercube(d=n_dim)
    samples = sampler.random(n=n_samples)
    scaled_samples = qmc.scale(samples, bounds[:, 0], bounds[:, 1])
    return scaled_samples

def fit_model(experimental_data, IPTG_ATC_conditions, scaled_samples):
    best_params = None
    best_error = float('inf')

    for i, params in enumerate(scaled_samples):
        error = objective_function(experimental_data, IPTG_ATC_conditions, params)
        if error < best_error:
            best_error = error
            best_params = params

    return best_params, best_error


# 读取参数
df = pd.read_excel(r"E:\250630\result\bistable steady params 260321\bsp166.xlsx")
for i, row in df.iterrows():
    globals()[row['Parameter']] = row['Value']
b_E, a_E, n_NE, k_NE = 1.71, 61.98, 0.73, 18.0
k_E, delta = 824.0, 0.013
n_IPTG, n_ATC, n_LT, n_TL =2.0, 2.0, 2.0, 2.0
b_T, a_T, k_LT, k_IPTG = b_T, a_T, k_LT, k_IPTG
b_L, a_L, k_TL, k_ATC = b_L, a_L, k_TL, k_ATC

SALI = 300.0
IPTG_ATC_conditions = [
    (0.0, 0.0), (0.0, 3.125), (0.0, 6.25), (0.0, 12.5),
    (25.0, 0.0), (25.0, 3.125), (25.0, 6.25), (25.0, 12.5),
    (50.0, 0.0), (50.0, 3.125), (50.0, 6.25), (50.0, 12.5),
    (100.0, 0.0), (100.0, 3.125), (100.0, 6.25), (100.0, 12.5)]

# m_E, m_T, m_L, d_E, d_T, d_L, n_E
n_dim = 7
n_samples = 1000
lower_bounds = [0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.5]
upper_bounds = [0.2, 0.4, 0.2, 0.2, 0.4, 0.2, 3.0]
bounds = np.column_stack((lower_bounds, upper_bounds))


# 开始学习
all_best_params = []
all_best_errors = []

for i in range(1000):
    if i%10==0:
        print(i)
    scaled_samples = lhsampling(n_dim,n_samples,bounds)
    best_params, best_error = fit_model(all_group_results_630, IPTG_ATC_conditions, scaled_samples)
    all_best_params.append(best_params.flatten())
    all_best_errors.append(best_error)

df = pd.DataFrame(all_best_params, columns=[f'param_{i+1}' for i in range(7)])
df['best_error'] = all_best_errors
result_dir = r"E:\250630\result"
df.to_excel(f'{result_dir}/best_dynamic_params_bsp166_1.xlsx', index=False)