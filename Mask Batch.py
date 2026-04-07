import os
import cv2
import numpy as np
import pandas as pd
from scipy.stats import qmc
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import minimize
from skimage.filters import threshold_li
from scipy.optimize import root, curve_fit
from ddeint import ddeint
from scipy.interpolate import interp1d


def process_image_pair(red_path, green_path, red_thresh=800, green_thresh=820):
    red = cv2.imread(red_path, cv2.IMREAD_UNCHANGED)
    green = cv2.imread(green_path, cv2.IMREAD_UNCHANGED)

    height, width = red.shape
    left = width // 3
    right = 2 * width // 3
    red = red[:, left:right]
    green = green[:, left:right]

    _, red_mask = cv2.threshold(red, threshold_li(red), 65535, cv2.THRESH_BINARY)
    _, green_mask = cv2.threshold(green, threshold_li(green), 65535, cv2.THRESH_BINARY)

    red_mask_bool = red_mask > 0
    green_mask_bool = green_mask > 0
    overlap_mask = red_mask_bool & green_mask_bool
    red_mask = red_mask_bool & ~overlap_mask
    green_mask = green_mask_bool & ~overlap_mask

    return {
        'red_in_red': float(red[red_mask].mean()),
        'green_in_red': float(green[red_mask].mean()),
        'red_in_green': float(red[green_mask].mean()),
        'green_in_green': float(green[green_mask].mean()),
        'red_mask_ratio': float(np.sum(red_mask) / red.size),
        'green_mask_ratio': float(np.sum(green_mask) / green.size),
        'overlap_ratio': float(np.sum(overlap_mask) / red.size)
    }

def process_position(position_dir):
    c2_dir = os.path.join(position_dir, 'c2')  # green
    c3_dir = os.path.join(position_dir, 'c3')  # red
    output_dir = os.path.join(position_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)

    green_files = sorted(f for f in os.listdir(c2_dir) if f.endswith('.tif'))
    red_files = sorted(f for f in os.listdir(c3_dir) if f.endswith('.tif'))
    results = []

    for green_file, red_file in zip(green_files, red_files):
        time_point = int(green_file.split('t')[1][:3])

        result = process_image_pair(
            os.path.join(c3_dir, red_file),
            os.path.join(c2_dir, green_file)
        )
        result['time_point'] = time_point
        results.append(result)

    pd.DataFrame(results).to_csv(
        os.path.join(output_dir, 'measurements.csv'),
        index=False
    )

    def batch_process(root_dir):
        for i in range(1, 65):  # xy01-xy64
            position_dir = os.path.join(root_dir, f'xy{i:02d}')
            process_position(position_dir)
            print(f"position{i} is done")

data_dir = r"E:\250630"
batch_process(data_dir)


def load_and_filter_data(root_dir, group_id, redbg, greenbg):
    start = group_id * 4 + 1
    end = start + 4

    BACKGROUNDS = {
        'red_in_red': redbg,
        'red_in_green': redbg,
        'green_in_red': greenbg,
        'green_in_green': greenbg}

    all_data = []

    for i in range(start, end):
        pos_dir = os.path.join(root_dir, f'xy{i:02d}', 'output', 'measurements.csv')
        df = pd.read_csv(pos_dir)

        # Filter noise
        cols_to_keep = ['time_point']
        if not (df['red_in_red'] < redbg * 2).all():
            cols_to_keep.extend(['red_in_red', 'green_in_red'])
        if not (df['green_in_green'] < greenbg * 2).all():
            cols_to_keep.extend(['red_in_green', 'green_in_green'])

        # Handle exceptions
        if i == 44:
            cols_to_keep = [col for col in cols_to_keep if col not in ['red_in_green', 'green_in_green']]

        # Subtract background
        corrected_df = df[cols_to_keep].copy()
        for col in corrected_df.columns:
            if col in BACKGROUNDS:
                corrected_df[col] = corrected_df[col] - BACKGROUNDS[col]

        all_data.append(corrected_df)

    if not all_data:
        return None

    # Combine four sets of data
    combined_df = pd.concat(all_data)
    grouped = combined_df.groupby('time_point').agg(['mean', 'std', 'count'])

    result = {}
    variables = ['red_in_red', 'red_in_green', 'green_in_red', 'green_in_green']

    for var in combined_df.columns:
        if var in variables:
            y_err = grouped[var]['std']
            if not y_err.isna().all() and not (y_err == 0).all():  # 判断是否只有一组函数，删除单组数据
                result[var] = {
                    'time_points': (grouped.index.values - 1) * 10,
                    'mean_values': grouped[var]['mean'].values,
                    'std_values': grouped[var]['std'].values}

    return result

def process_all_groups(root_dir, redbg, greenbg):
    all_results = []

    for group_id in range(16):  # 16 groups, 4 subgroups per group (xy01-xy64)
        group_data = load_and_filter_data(root_dir, group_id, redbg, greenbg)
        if group_data:
            all_results.append({
                'group_id': group_id,
                'data': group_data})

    return all_results

# Data format and usage
# '''
# all_results = [
#     {   # Group 0: (xy01-xy04)
#         'group_id': 0,
#         'data': {
#             'red_in_red': {'time_points': array, 'mean_values': array, 'std_values': array},
#             'red_in_green': {'time_points': array, 'mean_values': array, 'std_values': array},
#             'green_in_red': {'time_points': array, 'mean_values': array, 'std_values': array},
#             'green_in_green': {'time_points': array, 'mean_values': array, 'std_values': array}
#         }
#     },
#     {   # Group 1: (xy05-xy08)
#         'group_id': 1,
#         'data': {...}  # 同上
#     },
#     ...
#     # 16 sets in total
# ]

# # Get red channel data for Group 0
# group0 = all_group_results[0]
# red_data = group0['data']['red_in_red']
# time = red_data['time_points']  # time
# mean = red_data['mean_values']  # mean
# std = red_data['std_values']    # std
# print(f"time: {time}")
# print(f"mean: {mean}")
# print(f"std: {std}")
# '''

root_directory = r"E:\250630"
red_background = 542.4033861
green_background = 569.7855446

all_group_results_630 = process_all_groups(root_directory, red_background, green_background)


# save to excel
def save_to_excel(all_results, filename):
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        for i, group_data in enumerate(all_results):
            group_df = pd.DataFrame()
            for channel, values in group_data['data'].items():
                group_df[f'{channel}_time'] = values['time_points']
                group_df[f'{channel}_mean'] = values['mean_values']
                group_df[f'{channel}_std'] = values['std_values']
            group_df.to_excel(writer, sheet_name=f'group_{group_data["group_id"]}', index=False)

save_to_excel(all_group_results_630, 'E:/250630/all_group_results_630.xlsx')