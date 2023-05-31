import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, font_manager
from matplotlib.pyplot import subplots
from pandas import CategoricalDtype

houses = pd.read_excel('./../../results/faster_rcnn_complete_metric_real_data_different_conditions.xlsx')
houses = houses.loc[(houses['epochs_gd'] == 60) & ((houses['epochs_qd'] == 40) | (houses['epochs_qd'] == 60))]
houses = houses.loc[houses['dataset'] == 'gibson'].drop(['epochs_gd', 'epochs_qd', 'dataset'], axis=1)
houses_faster_rcnn = houses.groupby(['house', 'detector']).sum()


# YOLO
houses = pd.read_excel('./../../results/yolov5_complete_metric_real_data_different_condition.xlsx')
houses = houses.loc[(houses['epochs_gd'] == 60) & ((houses['epochs_qd'] == 40) | (houses['epochs_qd'] == 60))]
houses = houses.loc[houses['dataset'] == 'gibson'].drop(['epochs_gd', 'epochs_qd', 'dataset'], axis=1)
houses_yolo = houses.groupby(['house', 'detector']).sum()

# DETR
houses = pd.read_excel('./../../results/detr_complete_metrics_real_data_different_conditions.xlsx')
houses = houses.loc[(houses['epochs_gd'] == 60) & ((houses['epochs_qd'] == 40) | (houses['epochs_qd'] == 60))]
houses = houses.loc[houses['dataset'] == 'gibson'.upper()].drop(['epochs_gd', 'epochs_qd', 'dataset'], axis=1)
houses_detr = houses.groupby(['house', 'detector']).sum()


environments = ['floor1', 'floor4']
dataframes = [houses_faster_rcnn, houses_yolo, houses_detr]
labels = ['$GD_{-e}$', '$QD^{15}_e$', '$QD^{25}_e$', '$QD^{50}_e$', '$QD^{75}_e$']
experiments = ['GD', 'QD_15', 'QD_25', 'QD_50', 'QD_75']

table = ''
for i, env in enumerate(environments):
    table += '\multirow{5}{*}{$e_' + str((i+1)) + '$} &'
    for c, (label, exp) in enumerate(zip(labels, experiments)):
        if c > 0:
            table += ' & '
        total_positives = int(dataframes[0].loc[[(env, 'GD')]]['total_positives'][0])
        table += label + ' & ' + str(total_positives)
        TPs = [dataframe.loc[[(env, exp)]]['TP'][0] for dataframe in dataframes]
        FPs = [dataframe.loc[[(env, exp)]]['FP'][0] for dataframe in dataframes]
        FPious = [int(dataframe.loc[[(env, exp)]]['FPiou'][0]) for dataframe in dataframes]
        m = TPs.index(max(TPs))
        m1 = 0
        for e, element in enumerate(FPs):
            if int(element) == int(min(FPs)):
                m1 = e

        m2 = 0
        print(env, label)
        for e, element in enumerate(FPious):
            print(int(element), int(min(FPious)), m2)
            if int(element) == int(min(FPious)):
                m2 = e


        for d, dataframe in enumerate(dataframes):
            TP = int(dataframe.loc[[(env, exp)]]['TP'][0])
            FP = int(dataframe.loc[[(env, exp)]]['FP'][0])
            FPiou = int(dataframe.loc[[(env, exp)]]['FPiou'][0])

            TPstring = str(TP) + ' (' + str(int(TP / total_positives * 100)) + ')'
            if d == m:
                TPstring = '\\textbf{' + TPstring + '} '
            FPstring = str(FP) + ' (' + str(int(FP / total_positives * 100)) + ')'
            if d == m1:
                FPstring = '\\underline{' + FPstring + '} '

            FPiouString = str(FPiou) + ' (' + str(int(FPiou / total_positives * 100)) + ')'
            if d == m2:
                FPiouString = '\\underline{' + FPiouString + '} '
            table += ' & ' + TPstring + ' & ' + FPstring + ' & ' + FPiouString


        table += ' & ' + str(int(sum(TPs) / len(TPs))) + ' (' + str(int((sum(TPs) / len(TPs)) / total_positives * 100)) + ') & ' + str(int(sum(FPs) / len(FPs))) + ' (' + str(int((sum(FPs) / len(FPs)) / total_positives * 100)) + ') & ' + str(int(sum(FPious) / len(FPious))) + ' (' + str(int((sum(FPious) / len(FPious)) / total_positives * 100)) + ') '
        table += '\\\\ \n'
    table += '[2pt]\\hline \n'

print(table)