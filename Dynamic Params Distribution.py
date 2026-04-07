from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


base_path = r"E:\250630\result"

all_data = []
for i in range(1, 6):
    file_path = os.path.join(base_path, f"best_dynamic_params_bsp166_{i}.xlsx")
    df = pd.read_excel(file_path)
    all_data.append((df.iloc[:, :7].values, df['best_error'].values))

all_data.append
params_matrix = np.vstack([np.array([param.flatten() for param in data[0]]) for data in all_data])
error_data = np.hstack([data[1] for data in all_data])

print(f"参数矩阵形状: {params_matrix.shape}")
print(f"误差数据形状: {error_data.shape}")


fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()
param_names = ['m_E', 'm_T', 'm_L', 'd_E', 'd_T', 'd_L', 'n_E']

for i in range(7):
    axes[i].hist(params_matrix[:, i], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    # axes[i].axvline(params_matrix[:, i].mean(), color='red', linestyle='--', label=f'Mean: {params_matrix[:, i].mean():.3f}')
    axes[i].set_title(f'{param_names[i]}')
    axes[i].set_xlabel('Parameter Value')
    axes[i].set_ylabel('Frequency')
    # axes[i].locator_params(tight=True, nbins=6)
    axes[i].locator_params(axis='x', nbins=6)
    axes[i].locator_params(axis='y', nbins=6)
    # axes[i].legend()
    # axes[i].grid(True, alpha=0.3)

for i in range(7, 9):
    axes[i].set_visible(False)

plt.tight_layout()
plt.savefig(fr'E:\250630\result\dynamicparamsdistributionbsp166.pdf', format='pdf', bbox_inches='tight')
plt.show()

# axes[2].set_xlim(0.001, 0.02)
# axes[5].set_xlim(0.001, 0.02)


# 使用PCA降维到3维
pca = PCA(n_components=3)
pca_result = pca.fit_transform(params_matrix)
pc_means = np.mean(pca_result, axis=0)  # 三个主成分的均值
pc_stds = np.std(pca_result, axis=0)    # 三个主成分的标准差

kde = gaussian_kde(pca_result.T)
density = kde(pca_result.T)

# 最密的点
most_dense_idx = np.argmax(density)
most_dense_params = params_matrix[most_dense_idx]
most_dense_error = error_data[most_dense_idx]

# 选择密度最高的前20%的点
top20_percent = int(0.2 * len(density))
density_indices_sorted = np.argsort(density)[::-1]
top20_idx = density_indices_sorted[:top20_percent]
top20_error = error_data[top20_idx]
top20_density = density[top20_idx]

# 在这些高密度点中找出error最小的点
top20_minerror_idx = top20_idx[np.argmin(top20_error)]
top20_minerror_params = params_matrix[top20_minerror_idx]
top20_minerror_error = error_data[top20_minerror_idx]

# 全局最小误差点
global_minerror_idx = np.argmin(error_data)
global_minerror_params = params_matrix[global_minerror_idx]
global_minerror_error = error_data[global_minerror_idx]

# 创建3D图形
fig = plt.figure(figsize=(5,4))
ax = fig.add_subplot(111, projection='3d')

# 所有点
scatter_all = ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], c=density, alpha=0.4, cmap='PuBu', linewidths=0, s=10)

# top20点（用不同的标记和边缘颜色突出显示）
scatter_top20 = ax.scatter(pca_result[top20_idx, 0], pca_result[top20_idx, 1], pca_result[top20_idx, 2],
                          c=top20_density, alpha=1, cmap='PuBu', s=10,
                          edgecolors='black', linewidth=0.8, marker='o', depthshade=False)
# top20中error最小的点
ax.scatter(pca_result[top20_minerror_idx, 0], pca_result[top20_minerror_idx, 1], pca_result[top20_minerror_idx, 2],
           color='red', s=200, marker='*', edgecolors='white', linewidth=0.8, depthshade=False, label='Min Error in Top 20%')

# 全局最小误差点
# ax.scatter(pca_result[global_minerror_idx, 0], pca_result[global_minerror_idx, 1], pca_result[global_minerror_idx, 2],
#           color='red', s=200, marker='*', edgecolors='white', linewidth=0.5, zorder=100, label='Global Min Error')

# 最密的点
# ax.scatter(pca_result[most_dense_idx, 0], pca_result[most_dense_idx, 1], pca_result[most_dense_idx, 2],
#           color='blue', s=150, marker='s', edgecolors='white', linewidth=2, label='Most Dense')


cbar = plt.colorbar(scatter_all, ax=ax, pad=0.1)
cbar.set_label('Density')
ax.set_xlabel(f'PC1')
ax.set_ylabel(f'PC2')
ax.set_zlabel(f'PC3')
# 三轴等比例
ax.set_box_aspect([1,1,1])
# 对称范围
ax.set_xlim(pc_means[0] - 3*pc_stds[0], pc_means[0] + 3*pc_stds[0])
ax.set_ylim(pc_means[1] - 3*pc_stds[1], pc_means[1] + 3*pc_stds[1])
ax.set_zlim(pc_means[2] - 3*pc_stds[2], pc_means[2] + 3*pc_stds[2])
ax.xaxis.set_major_locator(MaxNLocator(5))
ax.yaxis.set_major_locator(MaxNLocator(5))
ax.zaxis.set_major_locator(MaxNLocator(5))
ax.legend(bbox_to_anchor=(1.05, 1), loc='best')
ax.view_init(elev=30, azim=45)  # 改变视角
plt.savefig(fr'E:\250630\result\dynamicparamsPCAbsp166.pdf', format='pdf', bbox_inches='tight')
plt.tight_layout()
plt.show()

# 打印统计信息
print(f"前3个主成分累计方差贡献率: {sum(pca.explained_variance_ratio_):.2%}")
print(f"PC1方差贡献率: {pca.explained_variance_ratio_[0]:.2%}")
print(f"PC2方差贡献率: {pca.explained_variance_ratio_[1]:.2%}")
print(f"PC3方差贡献率: {pca.explained_variance_ratio_[2]:.2%}")

print(f"\nmost_dense_idx: {most_dense_idx}")
print(f"most_dense_params: {most_dense_params}")
print(f"most_dense_error: {most_dense_error:.6f}")

print(f"\ntop20_minerror_idx: {top20_minerror_idx}")
print(f"top20_minerror_params: {top20_minerror_params}")
print(f"top20_minerror_error: {top20_minerror_error:.6f}")

print(f"\nglobal_minerror_idx: {global_minerror_idx}")
print(f"global_minerror_params: {global_minerror_params}")
print(f"global_minerror_error: {global_minerror_error:.6f}")


base_path = r"E:\250630\result"

all_data = []
for i in range(1, 6):
    file_path = os.path.join(base_path, f"best_dynamic_params_bsp166_{i}.xlsx")
    df = pd.read_excel(file_path)
    all_data.append((df.iloc[:, :7].values, df['best_error'].values))

all_data.append

params_matrix = np.vstack([np.array([param.flatten() for param in data[0]]) for data in all_data])
params_data = params_matrix[:, :6]
error_data = np.hstack([data[1] for data in all_data])
param_names = ['m_E', 'm_T', 'm_L', 'd_E', 'd_T', 'd_L']


new_order = ['m_E', 'm_T', 'm_L', 'd_E', 'd_T', 'd_L']

param_names = ['m_E', 'm_T', 'm_L', 'd_E', 'd_T', 'd_L']
col_indices = [param_names.index(name) for name in new_order]

top_50_params_reordered = params_data[:, col_indices]

correlation_matrix = pd.DataFrame(top_50_params_reordered).corr(method='spearman').values

# 创建坐标
fig, ax = plt.subplots(figsize=(4, 3))

for i in range(len(new_order)):
    for j in range(len(new_order)):
        corr_value = correlation_matrix[i, j]

        ax.scatter(j, i,
                   s=abs(corr_value) * 400,  # 点大小 = 相关系数绝对值
                   c=corr_value,  # 颜色 = 相关系数
                   cmap='coolwarm',
                   vmin=-1, vmax=1, zorder=100)

# 设置刻度
ax.set_xticks(range(len(new_order)))
ax.set_yticks(range(len(new_order)))
ax.set_xticklabels(new_order)
ax.set_yticklabels(new_order)

plt.colorbar(ax.collections[0], label='Spearman Correlation')
plt.gca().invert_yaxis()  # 让矩阵方向和heatmap一致
plt.grid(True)
plt.savefig(fr'E:\250630\result\dynamicparamscorrelationbsp166.pdf', format='pdf', bbox_inches='tight')
plt.tight_layout()
plt.show()


#记录top_params
result_dir = r"E:\250630\result"
param_names = ['m_E', 'm_T', 'm_L', 'd_E', 'd_T', 'd_L', 'n_E']

top_params_flat = top20_minerror_params.flatten() #展成一维
df = pd.DataFrame({'Parameter': param_names, 'Value': top_params_flat})
df.to_excel(f'{result_dir}/best_dynamic_params_bsp166_top20.xlsx', index=False)

dense_params_flat = most_dense_params.flatten() #展成一维
df = pd.DataFrame({'Parameter': param_names, 'Value': dense_params_flat})
df.to_excel(f'{result_dir}/best_dynamic_params_bsp166_dense.xlsx', index=False)

global_params_flat = global_minerror_params.flatten() #展成一维
df = pd.DataFrame({'Parameter': param_names, 'Value': global_params_flat})
df.to_excel(f'{result_dir}/best_dynamic_params_bsp166_global.xlsx', index=False)