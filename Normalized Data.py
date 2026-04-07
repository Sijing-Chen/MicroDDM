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


def visualize(experimental_data, IPTG_ATC_conditions):
    for i, (group, (IPTG, ATC)) in enumerate(zip(experimental_data, IPTG_ATC_conditions)):
        group_data = group['data']
        time_points = next(iter(group_data.values()))['time_points']

        # 判断数据状态
        has_set1 = 'red_in_red' in group_data and 'green_in_red' in group_data
        has_set2 = 'red_in_green' in group_data and 'green_in_green' in group_data
        ncol = 2 if (has_set1 and has_set2) else 1

        fig, axes = plt.subplots(1, ncol, figsize=(2.2, 1.65), sharey=True, squeeze=False, gridspec_kw={'wspace': 0})
        if ncol == 2:  # --- 两栏模式：左红(set1)，右绿(set2) ---
            # 左栏 (Set 1)
            axes[0, 0].errorbar(time_points, group_data['green_in_red']['mean_values'],
                                yerr=group_data['green_in_red']['std_values'], color='turquoise', fmt='o', alpha=0.7,
                                ms=0.8, capsize=1.0, capthick=0.5, elinewidth=0.3)
            axes[0, 0].errorbar(time_points, group_data['red_in_red']['mean_values'],
                                yerr=group_data['red_in_red']['std_values'], color='red', fmt='o', alpha=0.7, ms=0.8,
                                capsize=1.0, capthick=0.5, elinewidth=0.3)
            # 右栏 (Set 2)
            axes[0, 1].errorbar(time_points, group_data['green_in_green']['mean_values'],
                                yerr=group_data['green_in_green']['std_values'], color='darkgreen', fmt='^', alpha=0.7,
                                ms=0.8, capsize=1.0, capthick=0.5, elinewidth=0.3)
            axes[0, 1].errorbar(time_points, group_data['red_in_green']['mean_values'],
                                yerr=group_data['red_in_green']['std_values'], color='lightsalmon', fmt='^', alpha=0.7,
                                ms=0.8, capsize=1.0, capthick=0.5, elinewidth=0.3)
        else:  # --- 单栏模式：根据存在的数据绘制 ---
            ax = axes[0, 0]
            if has_set1:
                ax.errorbar(time_points, group_data['green_in_red']['mean_values'],
                            yerr=group_data['green_in_red']['std_values'], color='turquoise', fmt='o', alpha=0.7,
                            ms=0.8, capsize=1.0, capthick=0.5, elinewidth=0.5)
                ax.errorbar(time_points, group_data['red_in_red']['mean_values'],
                            yerr=group_data['red_in_red']['std_values'], color='red', fmt='o', alpha=0.7, ms=0.8,
                            capsize=1.0, capthick=0.5, elinewidth=0.5)
            elif has_set2:
                ax.errorbar(time_points, group_data['green_in_green']['mean_values'],
                            yerr=group_data['green_in_green']['std_values'], color='darkgreen', fmt='^', alpha=0.7,
                            ms=0.8, capsize=1.0, capthick=0.5, elinewidth=0.5)
                ax.errorbar(time_points, group_data['red_in_green']['mean_values'],
                            yerr=group_data['red_in_green']['std_values'], color='lightsalmon', fmt='^', alpha=0.7,
                            ms=0.8, capsize=1.0, capthick=0.5, elinewidth=0.5)

        # --- 样式统一调整 ---
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
        plt.savefig(fr'E:\250630\result\250630Data, IPTG={IPTG}, ATC={ATC}.pdf', format='pdf', bbox_inches='tight')
        plt.show()
        plt.close()


#实验条件
IPTG_ATC_conditions = [
    (0.0, 0.0), (0.0, 3.125), (0.0, 6.25), (0.0, 12.5),
    (25.0, 0.0), (25.0, 3.125), (25.0, 6.25), (25.0, 12.5),
    (50.0, 0.0), (50.0, 3.125), (50.0, 6.25), (50.0, 12.5),
    (100.0, 0.0), (100.0, 3.125), (100.0, 6.25), (100.0, 12.5)
]
SALI = 300.0

visualize(all_group_results_630, IPTG_ATC_conditions)