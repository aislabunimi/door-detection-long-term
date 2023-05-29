import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, font_manager
from matplotlib.pyplot import subplots
from pandas import CategoricalDtype

houses = pd.read_excel('./../../results/faster_rcnn_ap_simulation.xlsx')
houses['AP'] = houses['AP'].astype(np.float64)
houses['AP'] = houses['AP'].apply(lambda x: x*100).round()
houses = houses.loc[(houses['epochs_gd'] == 60) & ((houses['epochs_qd'] == 40) | (houses['epochs_qd'] == 60))]

labels = ['$GD_{-e}$', '$QD^{15}_e$', '$QD^{25}_e$', '$QD^{50}_e$', '$QD^{75}_e$']
experiments = ['GD', 'QD_15', 'QD_25', 'QD_50', 'QD_75']

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

closed_doors_faster_rcnn = closed_doors.drop([('house',)], axis=1)
open_doors_faster_rcnn = open_doors.drop([('house',)], axis=1)

increments_faster_rcnn = []
keys = pd.MultiIndex.from_arrays([['AP' for _ in range(len(experiments))], experiments])
for i in range(len(keys) -1):
    closed_doors_increment = (closed_doors.loc[:, keys[i + 1]] - closed_doors.loc[:, keys[i]]) / (closed_doors.loc[:, keys[i]]+1) * 100
    open_doors_increment = (open_doors.loc[:, keys[i + 1]] - open_doors.loc[:, keys[i]]) / (open_doors.loc[:, keys[i]] +1) * 100
    increments_faster_rcnn.append((closed_doors_increment, open_doors_increment))
    print(f'{keys[i + 1]}')
    print(f'\t- closed doors: mean = {closed_doors_increment.mean()}, std = {closed_doors_increment.std()}')
    print(f'\t- open doors: mean = {open_doors_increment.mean()}, std = {open_doors_increment.std()}')



# YOLO
houses = pd.read_excel('./../../results/yolov5_ap_simulation.xlsx')
houses['AP'] = houses['AP'].astype(np.float64)
houses['AP'] = houses['AP'].apply(lambda x: x*100).round()
houses = houses.loc[(houses['epochs_gd'] == 60) & ((houses['epochs_qd'] == 40) | (houses['epochs_qd'] == 60))]

labels = ['$GD_{-e}$', '$QD^{15}_e$', '$QD^{25}_e$', '$QD^{50}_e$', '$QD^{75}_e$']
experiments = ['GD', 'QD_15', 'QD_25', 'QD_50', 'QD_75']

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

closed_doors_faster_yolo = closed_doors.drop([('house',)], axis=1)
open_doors_faster_yolo = open_doors.drop([('house',)], axis=1)

increments_faster_yolo = []
keys = pd.MultiIndex.from_arrays([['AP' for _ in range(len(experiments))], experiments])
for i in range(len(keys) -1):
    closed_doors_increment = (closed_doors.loc[:, keys[i + 1]] - closed_doors.loc[:, keys[i]]) / closed_doors.loc[:, keys[i]] * 100
    open_doors_increment = (open_doors.loc[:, keys[i + 1]] - open_doors.loc[:, keys[i]]) / open_doors.loc[:, keys[i]] * 100
    increments_faster_yolo.append((closed_doors_increment, open_doors_increment))
    print(f'{keys[i + 1]}')
    print(f'\t- closed doors: mean = {closed_doors_increment.mean()}, std = {closed_doors_increment.std()}')
    print(f'\t- open doors: mean = {open_doors_increment.mean()}, std = {open_doors_increment.std()}')


# DETR

houses = pd.read_excel('./../../results/detr_ap_simulation.xlsx')
houses['AP'] = houses['AP'].astype(np.float64)
houses['AP'] = houses['AP'].apply(lambda x: x*100).round()
houses = houses.loc[(houses['epochs_gd'] == 60) & ((houses['epochs_qd'] == 40) | (houses['epochs_qd'] == 60))]

labels = ['$GD_{-e}$', '$QD^{15}_e$', '$QD^{25}_e$', '$QD^{50}_e$', '$QD^{75}_e$']
experiments = ['GD', 'QD_15', 'QD_25', 'QD_50', 'QD_75']

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

closed_doors_faster_detr = closed_doors.drop([('house',)], axis=1)
open_doors_faster_detr = open_doors.drop([('house',)], axis=1)

increments_faster_detr = []
keys = pd.MultiIndex.from_arrays([['AP' for _ in range(len(experiments))], experiments])
for i in range(len(keys) -1):
    closed_doors_increment = (closed_doors.loc[:, keys[i + 1]] - closed_doors.loc[:, keys[i]]) / closed_doors.loc[:, keys[i]] * 100
    open_doors_increment = (open_doors.loc[:, keys[i + 1]] - open_doors.loc[:, keys[i]]) / open_doors.loc[:, keys[i]] * 100
    increments_faster_detr.append((closed_doors_increment, open_doors_increment))
    print(f'{keys[i + 1]}')
    print(f'\t- closed doors: mean = {closed_doors_increment.mean()}, std = {closed_doors_increment.std()}')
    print(f'\t- open doors: mean = {open_doors_increment.mean()}, std = {open_doors_increment.std()}')

dataframes_closed = [closed_doors_faster_rcnn, closed_doors_faster_yolo, closed_doors_faster_detr]
increments = [increments_faster_rcnn, increments_faster_yolo, increments_faster_detr]

dataframes_open = [open_doors_faster_rcnn, open_doors_faster_yolo, open_doors_faster_detr]

table = ''
for i in range(5):
    table += '\multicolumn{1}{c|}{\multirow{2}{*}{' + labels[i] + '}} & Closed '
    m = [int(d.mean()[i]) for d in dataframes_closed].index(max([int(d.mean()[i]) for d in dataframes_closed]))

    for c, (d, inc) in enumerate(zip(dataframes_closed, increments)):
        if c != m:
            table += '& ' + str(int(np.rint(d.mean())[i])) + ' & ' + str(int(np.rint(d.std())[i])) + ' & '
        else:
            table += '& \\textbf{' + str(int(np.rint(d.mean())[i])) + '} & ' + str(int(np.rint(d.std())[i])) + ' & '
        if i == 0:
            table += ' -- & -- '
        else:
            table += '$' + str(int(np.rint(inc[i - 1][0].mean()))) + '\%$ & ' + str(int(np.rint(inc[i - 1][0].std())))

    table += '\\tabularnewline \n'

    table += '\multicolumn{1}{c|}{} & Open  '

    m = [int(d.mean()[i]) for d in dataframes_open].index(max([int(d.mean()[i]) for d in dataframes_open]))
    for c, (d, inc) in enumerate(zip(dataframes_open, increments)):
        if c != m:
            table += '& ' + str(int(np.rint(d.mean())[i])) + ' & ' + str(int(np.rint(d.std())[i])) + ' & '
        else:
            table += '& \\textbf{' + str(int(np.rint(d.mean())[i])) + '} & ' + str(int(np.rint(d.std())[i])) + ' & '

        if i == 0:
            table += ' -- & -- '
        else:
            table += '$' + str(int(np.rint(inc[i - 1][1].mean()))) + '\%$ & ' + str(int(np.rint(inc[i - 1][1].std())))

    table += '\\\\  \\hline \n'

print(table)