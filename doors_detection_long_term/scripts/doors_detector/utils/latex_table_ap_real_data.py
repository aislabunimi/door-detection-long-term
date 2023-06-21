import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, font_manager
from matplotlib.pyplot import subplots
from pandas import CategoricalDtype

# DETR

houses = pd.read_excel('./../../results/detr_ap_real_data_0.65.xlsx')
houses['AP'] = houses['AP'].astype(np.float64)
houses['AP'] = houses['AP'].apply(lambda x: x*100).round()
houses = houses.loc[(houses['dataset'] == 'GIBSON') & (houses['epochs_gd'] == 60) & ((houses['epochs_qd'] == 40) | (houses['epochs_qd'] == 60))]

labels = ['$GD_{-e}$', '$QD^{25}_e$', '$QD^{50}_e$', '$QD^{75}_e$']
experiments = ['GD', 'QD_25', 'QD_50', 'QD_75']

houses_list_dtype = CategoricalDtype(
    ['floor1', 'floor4', 'chemistry_floor0'],
    ordered=True)

closed_doors = houses[houses.label == 0][['house', 'detector', 'AP']]
closed_doors = closed_doors.pivot_table(values=['AP'], index=closed_doors['house'], columns='detector', aggfunc='first').reset_index()
closed_doors['house'] = closed_doors['house'].astype(houses_list_dtype)
closed_doors = closed_doors.sort_values(['house'])

open_doors = houses[houses.label == 1][['house', 'detector', 'AP']]
open_doors = open_doors.pivot_table(values=['AP'], index=open_doors['house'], columns='detector', aggfunc='first').reset_index()
open_doors['house'] = open_doors['house'].astype(houses_list_dtype)
open_doors = open_doors.sort_values(['house'])

closed_doors_detr_our = closed_doors.drop([('house',)], axis=1).drop([('AP', 'QD_15')], axis=1)
open_doors_faster_detr_out = open_doors.drop([('house',)], axis=1).drop([('AP', 'QD_15')], axis=1)

increments_detr_our = []
keys = pd.MultiIndex.from_arrays([['AP' for _ in range(len(experiments))], experiments])
for i in range(len(keys) -1):
    closed_doors_increment = (closed_doors.loc[:, keys[i + 1]] - closed_doors.loc[:, keys[i]]) / closed_doors.loc[:, keys[i]] * 100
    open_doors_increment = (open_doors.loc[:, keys[i + 1]] - open_doors.loc[:, keys[i]]) / open_doors.loc[:, keys[i]] * 100
    increments_detr_our.append((closed_doors_increment, open_doors_increment))
    print(f'{keys[i + 1]}')
    print(f'\t- closed doors: mean = {closed_doors_increment.mean()}, std = {closed_doors_increment.std()}')
    print(f'\t- open doors: mean = {open_doors_increment.mean()}, std = {open_doors_increment.std()}')


houses = pd.read_excel('./../../results/detr_ap_real_data_0.65.xlsx')
houses['AP'] = houses['AP'].astype(np.float64)
houses['AP'] = houses['AP'].apply(lambda x: x*100).round()
houses = houses.loc[(houses['dataset'] == 'DEEP_DOORS_2') & (houses['epochs_gd'] == 60) & ((houses['epochs_qd'] == 40) | (houses['epochs_qd'] == 60))]

labels = ['$GD_{-e}$', '$QD^{25}_e$', '$QD^{50}_e$', '$QD^{75}_e$']
experiments = ['GD', 'QD_25', 'QD_50', 'QD_75']

houses_list_dtype = CategoricalDtype(
    ['floor1', 'floor4', 'chemistry_floor0'],
    ordered=True)

closed_doors = houses[houses.label == 0][['house', 'detector', 'AP']]
closed_doors = closed_doors.pivot_table(values=['AP'], index=closed_doors['house'], columns='detector', aggfunc='first').reset_index()
closed_doors['house'] = closed_doors['house'].astype(houses_list_dtype)
closed_doors = closed_doors.sort_values(['house'])

open_doors = houses[houses.label == 1][['house', 'detector', 'AP']]
open_doors = open_doors.pivot_table(values=['AP'], index=open_doors['house'], columns='detector', aggfunc='first').reset_index()
open_doors['house'] = open_doors['house'].astype(houses_list_dtype)
open_doors = open_doors.sort_values(['house'])

closed_doors_detr_dd2 = closed_doors.drop([('house',)], axis=1).drop([('AP', 'QD_15')], axis=1)
open_doors_faster_detr_dd2 = open_doors.drop([('house',)], axis=1).drop([('AP', 'QD_15')], axis=1)

increments_detr_dd2 = []
keys = pd.MultiIndex.from_arrays([['AP' for _ in range(len(experiments))], experiments])
for i in range(len(keys) -1):
    closed_doors_increment = (closed_doors.loc[:, keys[i + 1]] - closed_doors.loc[:, keys[i]]) / closed_doors.loc[:, keys[i]] * 100
    open_doors_increment = (open_doors.loc[:, keys[i + 1]] - open_doors.loc[:, keys[i]]) / open_doors.loc[:, keys[i]] * 100
    increments_detr_dd2.append((closed_doors_increment, open_doors_increment))
    print(f'{keys[i + 1]}')
    print(f'\t- closed doors: mean = {closed_doors_increment.mean()}, std = {closed_doors_increment.std()}')
    print(f'\t- open doors: mean = {open_doors_increment.mean()}, std = {open_doors_increment.std()}')

table = ''
for i in range(4):
    table += '\multicolumn{1}{c|}{\multirow{2}{*}{' + labels[i] + '}} & Closed '

    for c, (d, inc) in enumerate(zip([closed_doors_detr_dd2, closed_doors_detr_our], [increments_detr_dd2, increments_detr_our])):
        table += '& ' + str(int(np.rint(d.mean())[i])) + ' & ' + str(int(np.rint(d.std())[i])) + ' & '
        if i == 0:
            table += ' -- & -- '
        else:
            table += '$' + str(int(np.rint(inc[i - 1][0].mean()))) + '\%$ & ' + str(int(np.rint(inc[i - 1][0].std())))

    table += '\\tabularnewline \n'

    table += '\multicolumn{1}{c|}{} & Open  '

    for c, (d, inc) in enumerate(zip([open_doors_faster_detr_dd2, open_doors_faster_detr_out], [increments_detr_dd2, increments_detr_our])):
        table += '& ' + str(int(np.rint(d.mean())[i])) + ' & ' + str(int(np.rint(d.std())[i])) + ' & '

        if i == 0:
            table += ' -- & -- '
        else:
            table += '$' + str(int(np.rint(inc[i - 1][1].mean()))) + '\%$ & ' + str(int(np.rint(inc[i - 1][1].std())))

    table += '\\\\  \\hline \n'

print(table)