import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, font_manager
from matplotlib.pyplot import subplots
from pandas import CategoricalDtype

houses = pd.read_excel('./../../../results/detr_ap_simulation_0.5.xlsx')
houses['AP'] = houses['AP'].astype(np.float64)
houses['AP'] = houses['AP'].apply(lambda x: x*100).round()
houses = houses.loc[(houses['epochs_gd'] == 60) & ((houses['epochs_qd'] == 40) | (houses['epochs_qd'] == 60))]

labels = ['$GD_{-e}$', '$QD^{25}_e$', '$QD^{50}_e$', '$QD^{75}_e$']
experiments = ['GD', 'QD_25', 'QD_50', 'QD_75']

houses_list_dtype = CategoricalDtype(
    ['house10', 'house2', 'house7', 'house9', 'house1', 'house13', 'house15', 'house20', 'house21', 'house22'],
    ordered=True)

closed_doors = houses[houses.label == 0][['house', 'detector', 'AP']]
closed_doors = closed_doors.pivot_table(values=['AP'], index=closed_doors['house'], columns='detector', aggfunc='first').reset_index()
closed_doors['house'] = closed_doors['house'].astype(houses_list_dtype)
closed_doors = closed_doors.sort_values(['house'])

open_doors = houses[houses.label == 1][['house', 'detector', 'AP']]
open_doors = open_doors.pivot_table(values=['AP'], index=open_doors['house'], columns='detector', aggfunc='first').reset_index()
open_doors['house'] = open_doors['house'].astype(houses_list_dtype)
open_doors = open_doors.sort_values(['house'])

fig, ax = subplots(figsize=(10, 5))

X = np.arange(11)
print([closed_doors[('AP', experiments[0])].mean() - closed_doors[('AP', experiments[0])].std(), closed_doors[('AP', experiments[0])].mean() + closed_doors[('AP', experiments[0])].std()])
ax.bar(X, closed_doors[('AP', experiments[0])].tolist() + [closed_doors[('AP', experiments[0])].mean()], yerr=np.array([[0, 0] for _ in range(10)] + [[closed_doors[('AP', experiments[0])].std(), closed_doors[('AP', experiments[0])].std()]]).T,width=0.2, label=labels[0])
ax.bar(X + 0.2, closed_doors[('AP', experiments[1])].tolist() + [closed_doors[('AP', experiments[1])].mean()], yerr=np.array([[0, 0] for _ in range(10)] + [[closed_doors[('AP', experiments[1])].std(), closed_doors[('AP', experiments[1])].std()]]).T,  width=0.2, label=labels[1])
ax.bar(X + 0.4, closed_doors[('AP', experiments[2])].tolist() + [closed_doors[('AP', experiments[2])].mean()], yerr=np.array([[0, 0] for _ in range(10)] + [[closed_doors[('AP', experiments[2])].std(), closed_doors[('AP', experiments[2])].std()]]).T, width=0.2, label=labels[2])
ax.bar(X + 0.6, closed_doors[('AP', experiments[3])].tolist() + [closed_doors[('AP', experiments[3])].mean()], yerr=np.array([[0, 0] for _ in range(10)] + [[closed_doors[('AP', experiments[3])].std(), closed_doors[('AP', experiments[3])].std()]]).T, width=0.2, label=labels[3])
ax.set_title('Closed doors', fontsize=18)
ax.set_ylim([0, 110])

ax.tick_params(axis='y', labelsize=16)
ax.set_xticks([i+0.3 for i in range(11)])
ax.set_xticklabels([f'$e_{i}$' for i in range(10)] + ['$\overline{e}$'], fontsize=17)
ax.set_ylabel('AP', fontsize=17)
ax.set_xlabel('Environment', fontsize=17)

ax.legend(prop={"size": 16}, loc='upper center', ncol=4)

fig.tight_layout()
plt.show()

plt.close()

fig, ax = subplots(figsize=(10, 5))

#print(closed_doors[('AP', '1')].tolist())

X = np.arange(11)
ax.bar(X, open_doors[('AP', experiments[0])].tolist() + [open_doors[('AP', experiments[0])].mean()], yerr=np.array([[0, 0] for _ in range(10)] + [[open_doors[('AP', experiments[0])].std(), open_doors[('AP', experiments[0])].std()]]).T, width=0.2, label=labels[0])
ax.bar(X + 0.2, open_doors[('AP', experiments[1])].tolist() + [open_doors[('AP', experiments[1])].mean()], yerr=np.array([[0, 0] for _ in range(10)] + [[open_doors[('AP', experiments[1])].std(), open_doors[('AP', experiments[1])].std()]]).T,  width=0.2, label=labels[1])
ax.bar(X + 0.4, open_doors[('AP', experiments[2])].tolist() + [open_doors[('AP', experiments[2])].mean()], yerr=np.array([[0, 0] for _ in range(10)] + [[open_doors[('AP', experiments[2])].std(), open_doors[('AP', experiments[2])].std()]]).T, width=0.2, label=labels[2])
ax.bar(X + 0.6, open_doors[('AP', experiments[3])].tolist() + [open_doors[('AP', experiments[3])].mean()], yerr=np.array([[0, 0] for _ in range(10)] + [[open_doors[('AP', experiments[3])].std(), open_doors[('AP', experiments[3])].std()]]).T, width=0.2, label=labels[3])

ax.set_title('Open doors', fontsize=18)
ax.set_ylim([0, 110])
ax.tick_params(axis='y', labelsize=16)
ax.set_xticks([i+0.3 for i in range(11)])
ax.set_xticklabels([f'$e_{i}$' for i in range(10)] + ['$\overline{e}$'], fontsize=17)
ax.set_ylabel('AP', fontsize=17)
ax.set_xlabel('Environment', fontsize=17)

ax.legend(prop={"size": 16}, loc='upper center', ncol=4)

fig.tight_layout()
plt.show()
closed_doors = closed_doors.drop([('house',)], axis=1)
open_doors = open_doors.drop([('house',)], axis=1)
closed_std = closed_doors.std()
closed_mean = closed_doors.mean()
open_std = open_doors.std()
open_mean = open_doors.mean()

print('closed_doors mean:', closed_mean)
print('closed_doors std:', closed_std)

print('open_doors mean:', open_mean)
print('open_std:', open_std)

increments = pd.DataFrame()
# Calculate increment

keys = pd.MultiIndex.from_arrays([['AP' for _ in range(len(experiments))], experiments])
for i in range(len(keys) -1):
    closed_doors_increment = (closed_doors.loc[:, keys[i + 1]] - closed_doors.loc[:, keys[i]]) / closed_doors.loc[:, keys[i]] * 100
    open_doors_increment = (open_doors.loc[:, keys[i + 1]] - open_doors.loc[:, keys[i]]) / open_doors.loc[:, keys[i]] * 100
    print(f'{keys[i + 1]}')
    print(f'\t- closed doors: mean = {closed_doors_increment.mean()}, std = {closed_doors_increment.std()}')
    print(f'\t- open doors: mean = {open_doors_increment.mean()}, std = {open_doors_increment.std()}')
