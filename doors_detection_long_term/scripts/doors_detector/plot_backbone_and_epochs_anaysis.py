import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.pyplot import subplots

results = pd.read_excel('./../results/epochsand_backbone_aggregation.xlsx', index_col=[0,1,2,3,4])


labels = ['$GD_{-e}$', '$QD^{25}_e$', '$QD^{50}_e$', '$QD^{75}_e$']

val = results.loc[('2_layers', 'GD', 40, 40, 1), 'AP']
fig, ax = subplots(figsize=(10, 5))

for backbone, epochs_general, epochs in [(b, e_g, e) for b in ['fixed', '2_layers'] for e_g in [40, 60] for e in [20, 40]]:
    y = [results.loc[(backbone, 'GD',epochs_general, epochs_general, 0), 'AP']] + [results.loc[(backbone, d,epochs_general, epochs, 0), 'AP'] for d in ['QD_25', 'QD_50', 'QD_75']]
    ax.plot([0, 1, 2, 3], y, label=f'Backbone {backbone}, {epochs_general} epochs GD, {epochs} epochs')



ax.set_title('AP results - closed doors', fontsize=18)
ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()
plt.show()
plt.close()

fig, ax = subplots(figsize=(10, 5))

for backbone, epochs_general, epochs in [(b, e_g, e) for b in ['fixed', '2_layers'] for e_g in [40, 60] for e in [20, 40]]:
    y = [results.loc[(backbone, 'GD',epochs_general, epochs_general, 1), 'AP']] + [results.loc[(backbone, d,epochs_general, epochs, 1), 'AP'] for d in ['QD_25', 'QD_50', 'QD_75']]
    ax.plot([0, 1, 2, 3], y, label=f'Backbone {backbone}, {epochs_general} epochs GD, {epochs} epochs')



ax.set_title('AP results - open doors', fontsize=18)
ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()
plt.show()
plt.close()