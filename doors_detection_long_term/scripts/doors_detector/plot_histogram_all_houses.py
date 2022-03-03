import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

houses = pd.read_csv('./../results/risultati_tesi_antonazzi.csv')
houses['AP'] = houses['AP'].astype(np.float64)

print(houses.dtypes)
houses['AP'] = houses['AP'].apply(lambda x: x*100).round()

houses_list = houses['Env name'].unique().tolist()

h = houses['Exp']
houses_minus_1 = houses.loc[houses['Exp'] == '-1']


grouped = houses.groupby('Env name')
dataframes = [grouped.get_group(name) for name in houses_list]

print(grouped.get_group('house1'))
print(houses)

# Computes closed door graph

fig, ax = plt.subplots(figsize=(8, 5))

dataframes.sort(key=lambda x: x['AP'].iloc[4])
y = [dataframes[i]['AP'].iloc[1] for i in range(10)]
houses_list = [dataframes[i]['Env name'].iloc[1] for i in range(10)]
labels = ['$GD$', '$FD_{25}$', '$FD_{50}$', '$FD_{75}$']
markers = ['^', 'D', 's', 'x']
for i in range(4):

    ax.plot([i*2 for i in range(10)], [dataframes[z]['AP'].iloc[1 + i * 3] for z in range(10)], label=labels[i], marker=markers[i])


ax.set_ylabel('AP')
ax.set_ylim([0, 110])
ax.set_title('AP results over all houses - closed doors (0)')
ax.set_xticks([i*2 for i in range(10)])
ax.set_xticklabels(houses_list)
ax.legend()

fig.tight_layout()

plt.show()

plt.close()

# Plot open doors
fig, ax = plt.subplots(figsize=(8, 5))

dataframes.sort(key=lambda x: x['AP'].iloc[5])
y = [dataframes[i]['AP'].iloc[1] for i in range(10)]
houses_list = [dataframes[i]['Env name'].iloc[1] for i in range(10)]
labels = ['$GD$', '$FD_{25}$', '$FD_{50}$', '$FD_{75}$']
markers = ['^', 'D', 's', 'x']
for i in range(4):

    ax.plot([i*2 for i in range(10)], [dataframes[z]['AP'].iloc[2 + i * 3] for z in range(10)], label=labels[i], marker=markers[i])


ax.set_ylabel('AP')
ax.set_ylim([0, 110])
ax.set_title('AP results over all houses - open doors (1)')
ax.set_xticks([i*2 for i in range(10)])
ax.set_xticklabels(houses_list)
ax.legend()

fig.tight_layout()

plt.show()

# Plot open doors
fig, ax = plt.subplots(figsize=(8, 5))

dataframes.sort(key=lambda x: x['AP'].iloc[5])
y = [dataframes[i]['AP'].iloc[1] for i in range(10)]
houses = [dataframes[i]['Env name'].iloc[1] for i in range(10)]
labels = ['$GD$', '$FD_{25}$', '$FD_{50}$', '$FD_{75}$']
markers = ['^', 'D', 's', 'x']
ticks = []
for i in range(4):
    y = [dataframes[z]['AP'].iloc[2 + i * 3] for z in range(10)]
    ticks += y
    ax.plot([i*2 for i in range(10)], y, label=labels[i], marker=markers[i])


ax.set_ylabel('AP')
ax.set_ylim([0, 110])
ax.set_title('AP results over all houses - open doors (1)')
#ax.set_yticks(ticks)
ax.set_xticks([i*2 for i in range(10)])
#ax.set_xticklabels(ax.get_xticks(), rotation = 45)
ax.set_xticklabels(houses_list)
ax.legend()

fig.tight_layout()

plt.show()





