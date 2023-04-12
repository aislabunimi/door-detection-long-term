import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.pyplot import subplots

results = pd.read_excel('./../../results/general_detectors_with_real_data.xlsx', index_col=[1,2,3,4,5])


labels = ['40', '60', '80', '100']

#val = results.loc[('2_layers', 'GD', 40, 40, 1), 'AP']
fig, ax = subplots(figsize=(10, 5))

for backbone, train_dataset in [(b, t_d) for b in ['2_LAYERS', 'FIXED'] for t_d in ['GIBSON', 'DEEP_DOORS_2', 'GIBSON_DEEP_DOORS_2_HALF', 'GIBSON_DEEP_DOORS_2']]:
    epochs_general = [40, 60]
    if backbone == '2_LAYERS':
        epochs_general += [80, 100]

    y = [results.loc[('floor4', backbone, train_dataset, epochs, 0), 'AP'] for epochs in epochs_general]
    ax.plot([i for  i in range(len(epochs_general))], y, '--' if backbone == 'FIXED' else '-', label=f'Backbone: {backbone}, train dataset: {train_dataset}')



ax.set_title('AP results - closed doors', fontsize=18)
ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()
plt.show()
plt.close()


fig, ax = subplots(figsize=(10, 5))

for backbone, train_dataset in [(b, t_d) for b in ['2_LAYERS', 'FIXED'] for t_d in ['GIBSON', 'DEEP_DOORS_2', 'GIBSON_DEEP_DOORS_2_HALF', 'GIBSON_DEEP_DOORS_2']]:
    epochs_general = [40, 60]
    if backbone == '2_LAYERS':
        epochs_general += [80, 100]

    y = [results.loc[('floor4', backbone, train_dataset, epochs, 1), 'AP'] for epochs in epochs_general]
    ax.plot([i for  i in range(len(epochs_general))], y, '--' if backbone == 'FIXED' else '-', label=f'Backbone: {backbone}, train dataset: {train_dataset}')



ax.set_title('AP results - open doors', fontsize=18)
ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()
plt.show()
plt.close()