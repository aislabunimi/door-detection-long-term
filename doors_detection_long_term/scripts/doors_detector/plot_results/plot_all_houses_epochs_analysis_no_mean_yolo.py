import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, font_manager
from matplotlib.pyplot import subplots
from pandas import CategoricalDtype

houses = pd.read_excel('./../../results/yolo_v5_epochs_analysis.xlsx')
houses['AP'] = houses['AP'].astype(np.float64)
houses['AP'] = houses['AP'].apply(lambda x: x*100).round()
houses = houses.loc[(houses['epochs_gd'] == 60) & ((houses['epochs'] == 40) | (houses['epochs'] == 60))]

labels = ['$GD_{-e}$', '$QD^{25}_e$', '$QD^{50}_e$', '$QD^{75}_e$']
experiments = ['GD', 'QD_25', 'QD_50', 'QD_75']

houses_list_dtype = CategoricalDtype(
    ['house10', 'house2', 'house7', 'house9', 'house1', 'house13', 'house15', 'house20', 'house21', 'house22'],
    ordered=True
)

closed_doors = houses[houses.label == 0][['house', 'detector', 'AP']]
closed_doors = closed_doors.pivot_table(values=['AP'], index=closed_doors['house'], columns='detector', aggfunc='first').reset_index()
closed_doors['house'] = closed_doors['house'].astype(houses_list_dtype)
closed_doors = closed_doors.sort_values(['house'])

open_doors = houses[houses.label == 1][['house', 'detector', 'AP']]
open_doors = open_doors.pivot_table(values=['AP'], index=open_doors['house'], columns='detector', aggfunc='first').reset_index()
open_doors['house'] = open_doors['house'].astype(houses_list_dtype)
open_doors = open_doors.sort_values(['house'])

fig, ax = subplots(figsize=(10, 5))

X = np.arange(10)

ax.bar(X, closed_doors[('AP', experiments[0])].tolist(),width=0.2, label=labels[0])
ax.bar(X + 0.2, closed_doors[('AP', experiments[1])].tolist(),  width=0.2, label=labels[1])
ax.bar(X + 0.4, closed_doors[('AP', experiments[2])].tolist(), width=0.2, label=labels[2])
ax.bar(X + 0.6, closed_doors[('AP', experiments[3])].tolist(), width=0.2, label=labels[3])

ax.set_title('Closed doors', fontsize=18)
ax.set_ylim([0, 110])

ax.tick_params(axis='y', labelsize=16)
ax.set_xticks([i+0.3 for i in range(10)])
ax.set_xticklabels([f'$e_{i}$' for i in range(10)], fontsize=17)
ax.set_ylabel('AP', fontsize=17)
ax.set_xlabel('Environment', fontsize=17)

ax.legend(prop={"size": 16}, loc='upper center', ncol=4)

fig.tight_layout()
plt.show()

plt.close()

fig, ax = subplots(figsize=(10, 5))

#print(closed_doors[('AP', '1')].tolist())

X = np.arange(10)
ax.bar(X, open_doors[('AP', experiments[0])].tolist(), width=0.2, label=labels[0])
ax.bar(X + 0.2, open_doors[('AP', experiments[1])].tolist(),  width=0.2, label=labels[1])
ax.bar(X + 0.4, open_doors[('AP', experiments[2])].tolist(), width=0.2, label=labels[2])
ax.bar(X + 0.6, open_doors[('AP', experiments[3])].tolist(), width=0.2, label=labels[3])

ax.set_title('Open doors', fontsize=18)
ax.set_ylim([0, 110])
ax.tick_params(axis='y', labelsize=16)
ax.set_xticks([i+0.3 for i in range(10)])
ax.set_xticklabels([f'$e_{i}$' for i in range(10)], fontsize=17)
ax.set_ylabel('AP', fontsize=17)
ax.set_xlabel('Environment', fontsize=17)
#ax.set_yticks([])

ax.legend(prop={"size": 16}, loc='upper center', ncol=4)

fig.tight_layout()
plt.show()

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
for i, exp in enumerate(experiments[1:]):
    i += 1
    closed_doors_increment = (closed_doors.iloc[:, i + 1] - closed_doors.iloc[:, i]) / closed_doors.iloc[:, i] * 100
    open_doors_increment = (open_doors.iloc[:, i + 1] - open_doors.iloc[:, i]) / open_doors.iloc[:, i] * 100
    print(f'{exp}')
    print(f'\t- closed doors: mean = {closed_doors_increment.mean()}, std = {closed_doors_increment.std()}')
    print(f'\t- open doors: mean = {open_doors_increment.mean()}, std = {open_doors_increment.std()}')