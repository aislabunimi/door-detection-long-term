import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, font_manager
from matplotlib.pyplot import subplots
from pandas import CategoricalDtype

houses = pd.read_csv('./../results/risultati_tesi_antonazzi.csv')
houses['AP'] = houses['AP'].astype(np.float64)
houses['AP'] = houses['AP'].apply(lambda x: x*100).round()

labels = ['$GD_{-e}$', '$QD^{25}_e$', '$QD^{50}_e$', '$QD^{75}_e$']
experiments = ['1', '2a', '2b', '2c']

houses_list_dtype = CategoricalDtype(
    ['house1', 'house2', 'house7', 'house9', 'house10', 'house13', 'house15', 'house20', 'house21', 'house22'],
    ordered=True
)

closed_doors = houses[houses.Label == 0][['Env name', 'Exp', 'AP']]
closed_doors = closed_doors.pivot_table(values=['AP'], index=closed_doors['Env name'], columns='Exp', aggfunc='first').reset_index()
closed_doors['Env name'] = closed_doors['Env name'].astype(houses_list_dtype)
closed_doors = closed_doors.sort_values(['Env name'])

open_doors = houses[houses.Label == 1][['Env name', 'Exp', 'AP']]
open_doors = open_doors.pivot_table(values=['AP'], index=open_doors['Env name'], columns='Exp', aggfunc='first').reset_index()
open_doors['Env name'] = open_doors['Env name'].astype(houses_list_dtype)
open_doors = open_doors.sort_values(['Env name'])

fig, ax = subplots(figsize=(10, 5))

X = np.arange(11)

ax.bar(X, closed_doors[('AP', experiments[0])].tolist() + [closed_doors[('AP', experiments[0])].mean()], width=0.2, label=labels[0])
ax.bar(X + 0.2, closed_doors[('AP', experiments[1])].tolist() + [closed_doors[('AP', experiments[1])].mean()],  width=0.2, label=labels[1])
ax.bar(X + 0.4, closed_doors[('AP', experiments[2])].tolist() + [closed_doors[('AP', experiments[2])].mean()],  width=0.2, label=labels[2])
ax.bar(X + 0.6, closed_doors[('AP', experiments[3])].tolist() + [closed_doors[('AP', experiments[3])].mean()],  width=0.2, label=labels[3])

ax.set_title('AP results over all houses - closed doors', fontsize=18)
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

print(closed_doors[('AP', '1')].tolist())

X = np.arange(11)
ax.bar(X, open_doors[('AP', experiments[0])].tolist() + [open_doors[('AP', experiments[0])].mean()], width=0.2, label=labels[0])
ax.bar(X + 0.2, open_doors[('AP', experiments[1])].tolist() + [open_doors[('AP', experiments[1])].mean()],  width=0.2, label=labels[1])
ax.bar(X + 0.4, open_doors[('AP', experiments[2])].tolist() + [open_doors[('AP', experiments[2])].mean()],  width=0.2, label=labels[2])
ax.bar(X + 0.6, open_doors[('AP', experiments[3])].tolist() + [open_doors[('AP', experiments[3])].mean()],  width=0.2, label=labels[3])

ax.set_title('AP results over all houses - open doors', fontsize=18)
ax.set_ylim([0, 110])
ax.tick_params(axis='y', labelsize=16)
ax.set_xticks([i+0.3 for i in range(11)])
ax.set_xticklabels([f'$e_{i}$' for i in range(10)] + ['$\overline{e}$'], fontsize=17)
ax.set_ylabel('AP', fontsize=17)
ax.set_xlabel('Environment', fontsize=17)

ax.legend(prop={"size": 16}, loc='upper center', ncol=4)

fig.tight_layout()
plt.show()