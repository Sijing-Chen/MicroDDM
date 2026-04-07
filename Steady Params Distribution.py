import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_excel(r"E:\250630\result\best_params_1000_iterations_1110.xlsx")
params_data = df.iloc[:, :9].values
error_data = df['best_error'].values
params_matrix = np.array([param.flatten() for param in params_data])

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()
param_names = ['k_IPTG', 'k_ATC', 'k_LT', 'k_TL', 'b_T', 'b_L', 'a_T', 'a_L', 'n_E']

for i in range(9):
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

plt.tight_layout()
plt.savefig(fr'E:\250630\result\steadyparamsdistribution.pdf', format='pdf', bbox_inches='tight')
plt.show()