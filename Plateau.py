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


def hill_decay(t, a, k, n, plateau):
    return plateau - (plateau - a) * (t**n / (k**n + t**n))

def safe_fit(time, mean):
    try:
        params, _ = curve_fit(hill_decay,time,mean, bounds=([-np.inf, 0, -np.inf, 0], [np.inf, np.inf, 0, np.inf]))#,p0=[0, 10, 2, 1],maxfev=5000
        plateau = params[-1]
        if plateau > 10000 or np.isnan(plateau) or plateau > np.nanmax(mean):
            raise RuntimeError("Plateau value out of range") #手动引发异常，进入except
        return plateau, True
    except:
        last_10pct = int(len(time) * 0.1)
        return np.nanmean(mean[-last_10pct:]), False


def fit_all_groups(all_results):
    var_config = {
        'red_in_red': {'color': 'red', 'marker': 'o', 'col_idx': 0},
        'green_in_red': {'color': 'turquoise', 'marker': 'o', 'col_idx': 0},
        'red_in_green': {'color': 'lightsalmon', 'marker': '^', 'col_idx': 1},
        'green_in_green': {'color': 'darkgreen', 'marker': '^', 'col_idx': 1}
    }

    result_dir = os.path.join(root_directory, 'result')
    os.makedirs(result_dir, exist_ok=True)
    plateau_df = pd.DataFrame(columns=['Group', 'Variable', 'Plateau'])

    for group in all_results:
        group_id = group['group_id']
        data = group['data']

        # 1. 确定分栏数量
        has_set1 = 'red_in_red' in data and 'green_in_red' in data
        has_set2 = 'red_in_green' in data and 'green_in_green' in data
        ncol = 2 if (has_set1 and has_set2) else 1

        # 2. 创建画布
        fig, axes = plt.subplots(1, ncol, figsize=(2.2, 1.65), sharey=True, squeeze=False, gridspec_kw={'wspace': 0})

        # 3. 遍历变量并绘图
        for var_name, var_data in data.items():
            cfg = var_config[var_name]
            time = var_data['time_points']
            mean = var_data['mean_values']

            # 分栏路由：单栏去0，双栏按配置去0或1
            target_ax = axes[0, cfg['col_idx'] if ncol == 2 else 0]

            # 拟合
            plateau, is_fitted = safe_fit(time, mean)
            plateau_df.loc[len(plateau_df)] = [f'Group {group_id + 1}', var_name, plateau]

            # 绘制原始数据
            target_ax.scatter(time, mean, color=cfg['color'], marker=cfg['marker'], s=5.0)

            # 绘制平台线
            target_ax.axhline(plateau, color=cfg['color'], linestyle='-.', linewidth=1.8)

            # # 绘制拟合曲线
            # if is_fitted:
            #     params, _ = curve_fit(hill_decay, time, mean, bounds=([-np.inf, 0, -np.inf, 0], [np.inf, np.inf, 0, np.inf]))
            #     fit_curve = hill_decay(time, *params)
            #     target_ax.plot(time, fit_curve, color=cfg['color'], linewidth=0.8)

        # 4. 样式调整（保持之前的右侧刻度风格）
        for j, ax in enumerate(axes.flatten()):
            # ax.set_xlim(0, 1100)
            # ax.set_ylim(0, 2.0)
            ax.tick_params(axis='both', which='major', labelsize=7)
            ax.locator_params(axis='x', nbins=5)
            ax.locator_params(axis='y', nbins=5)

            if j == ncol - 1:
                ax.yaxis.tick_right()
                ax.yaxis.set_label_position("right")
                ax.tick_params(axis='y', which='both', right=True, labelright=True, left=False, labelleft=False)
            else:
                ax.tick_params(axis='y', which='both', left=False, labelleft=False)

        # 5. 保存与展示
        plt.tight_layout()
        plt.savefig(f'{result_dir}/Group {group_id + 1} plateau.pdf', format='pdf', bbox_inches='tight')
        plt.show()
        plt.close()

    plateau_df.to_excel(f'{result_dir}/plateau260225.xlsx', index=False)


def plot_single_data(group_id):
    group_id -= 1

    def re_fit(time, mean):
        last_10pct = int(len(time) * 0.1)
        return np.nanmean(mean[-last_10pct:])

    # 路由配置：0进左栏，1进右栏
    var_config = {
        'red_in_red': {'color': 'red', 'marker': 'o', 'col_idx': 0},
        'green_in_red': {'color': 'turquoise', 'marker': 'o', 'col_idx': 0},
        'red_in_green': {'color': 'lightsalmon', 'marker': '^', 'col_idx': 1},
        'green_in_green': {'color': 'darkgreen', 'marker': '^', 'col_idx': 1}
    }

    result_dir = os.path.join(root_directory, 'result')
    excel_path = f'{result_dir}/plateau260225.xlsx'
    plateau_df = pd.read_excel(excel_path)

    data = all_group_results[group_id]['data']

    has_set1 = 'red_in_red' in data and 'green_in_red' in data
    has_set2 = 'red_in_green' in data and 'green_in_green' in data
    ncol = 2 if (has_set1 and has_set2) else 1

    fig, axes = plt.subplots(1, ncol, figsize=(2.2, 1.65), sharey=True, squeeze=False, gridspec_kw={'wspace': 0})

    for var_name, var_data in data.items():
        cfg = var_config[var_name]
        time = var_data['time_points']
        mean = var_data['mean_values']

        # 计算并更新 Excel 中的 Plateau
        plateau = re_fit(time, mean)
        plateau_df.loc[(plateau_df['Group'] == f'Group {group_id + 1}') &
                       (plateau_df['Variable'] == var_name), 'Plateau'] = plateau

        # 分栏路由
        target_ax = axes[0, cfg['col_idx'] if ncol == 2 else 0]

        # 绘制原始数据
        target_ax.scatter(time, mean, color=cfg['color'], marker=cfg['marker'], s=5.0)
        # 绘制平台线
        target_ax.axhline(plateau, color=cfg['color'], linestyle='-.', linewidth=1.8)

    # 保存更新后的 Excel
    plateau_df.to_excel(excel_path, index=False)

    for j, ax in enumerate(axes.flatten()):
        # ax.set_xlim(0, 1100)
        # ax.set_ylim(0, 2.0)
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
    plt.savefig(f'{result_dir}/Group {group_id + 1} plateau.pdf', format='pdf', bbox_inches='tight')
    plt.show()
    plt.close()


fit_all_groups(all_group_results)
plot_single_data(10)
plot_single_data(11)
plot_single_data(15)
