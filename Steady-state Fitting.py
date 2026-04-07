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


file_path = r"E:\250630\result\steadystate.xlsx"
df = pd.read_excel(file_path)

data = df.to_numpy()

print(data.shape)
print(data)


def model_equations(vars, IPTG, ATC, params):
    L, T = vars
    k_IPTG, k_ATC, k_LT, k_TL, b_T, b_L, a_T, a_L, n_E = params

    f_IPTG = k_IPTG ** n_IPTG / (IPTG ** n_IPTG + k_IPTG ** n_IPTG)
    f_ATC = k_ATC ** n_ATC / (ATC ** n_ATC + k_ATC ** n_ATC)
    term_E = E ** n_E / (E ** n_E + k_E ** n_E)

    eq1 = L - (b_L / delta + a_L / delta * (k_TL ** n_TL / ((f_ATC * T) ** n_TL + k_TL ** n_TL)) * term_E)
    eq2 = T - (b_T / delta + a_T / delta * (k_LT ** n_LT / ((f_IPTG * L) ** n_LT + k_LT ** n_LT)) * term_E)

    return [eq1, eq2]


def solve_steady_state(IPTG, ATC, params):
    result = root(model_equations, [2000, 2000], args=(IPTG, ATC, params), method='hybr')
    if result.success:
        return result.x
    else:
        return [np.nan, np.nan]


def objective_function(data, params):
    total_error = 0
    for i in range(len(data)):
        IPTG, ATC, L_exp, T_exp = data[i]
        L_pred, T_pred = solve_steady_state(IPTG, ATC, params)

        if not np.isnan(L_pred) and not np.isnan(T_pred):
            error_L = (L_pred - L_exp) ** 2
            error_T = (T_pred - T_exp) ** 2
            total_error += error_L + error_T
        else:
            total_error += 1e10
    return total_error


def lhsampling(n_dim,n_samples,bounds):
    sampler = qmc.LatinHypercube(d=n_dim)
    samples = sampler.random(n=n_samples)
    scaled_samples = qmc.scale(samples, bounds[:, 0], bounds[:, 1])
    return scaled_samples


def fit_model(data, scaled_samples):
    best_params = None
    best_error = float('inf')
    all_results = []

    for i, params in enumerate(scaled_samples):
        error = objective_function(data, params)
        all_results.append({'params': params, 'error': error})
        if error < best_error:
            best_error = error
            best_params = params
            # print(f"Sample {i+1}/{len(scaled_samples)} - Error: {error:.4f}")
    all_results.sort(key=lambda x: x['error'])
    return best_params, best_error, all_results


#固定参数
b_E = 1.71
a_E = 61.98
n_NE = 0.73
k_NE = 18.0
k_E = 824.0
delta = 0.013
n_IPTG = 2.0
n_ATC = 2.0
n_LT = 2.0
n_TL = 2.0
SALI = 300.0
E = b_E/delta + a_E/delta * (SALI**n_NE/(SALI**n_NE+k_NE**n_NE))


n_dim = 9
n_samples = 10000
#[k_IPTG, k_ATC, k_LT, k_TL,
# b_T, b_L, a_T, a_L,
# n_E]
lower_bounds = [0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
                0.0]
upper_bounds = [100.0, 20.0, 800.0, 800.0,
                20.0, 20.0, 100.0, 100.0,
                4.0]
bounds = np.column_stack((lower_bounds, upper_bounds))
scaled_samples = lhsampling(n_dim,n_samples,bounds)
print(scaled_samples.shape)


all_best_params = []
all_best_errors = []

for i in range(1000):
    if i%10==0:
        print(i)
    scaled_samples = lhsampling(n_dim,n_samples,bounds)
    best_params, best_error, all_results = fit_model(data, scaled_samples)
    all_best_params.append(best_params.flatten())
    all_best_errors.append(best_error)

df = pd.DataFrame(all_best_params, columns=[f'param_{i+1}' for i in range(9)])
df['best_error'] = all_best_errors
result_dir = r"E:\250630\result"
df.to_excel(f'{result_dir}/best_params_1000_iterations_1110.xlsx', index=False)