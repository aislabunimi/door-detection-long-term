import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, font_manager
from matplotlib.pyplot import subplots
from pandas import CategoricalDtype

# DETR
houses = pd.read_excel('./../../results/detr_complete_metrics_real_data_0.5.xlsx')
houses = houses.loc[(houses['epochs_gd'] == 60) & ((houses['epochs_qd'] == 40) | (houses['epochs_qd'] == 60))]
houses = houses.loc[houses['dataset'] == 'gibson'.upper()].drop(['epochs_gd', 'epochs_qd', 'dataset'], axis=1)
houses_detr = houses.groupby(['house', 'detector']).sum()

dataframes = [houses_detr]
environments = ['floor1', 'floor4', 'chemistryfloor0']
labels = ['$GD_{-e}$', '$QD^{25}_e$', '$QD^{50}_e$', '$QD^{75}_e$']
experiments = ['GD', 'QD_25', 'QD_50', 'QD_75']

table = ''
for i, env in enumerate(environments):
    table += '\multirow{4}{*}{$e_' + str((i+1)) + '$} &'
    for c, (label, exp) in enumerate(zip(labels, experiments)):
        if c > 0:
            table += ' & '
        total_positives = int(dataframes[0].loc[[(env, 'GD')]]['total_positives'][0])
        table += label + ' & ' + str(total_positives)
        TP = houses_detr.loc[[(env, exp)]]['TP'][0]
        FP = houses_detr.loc[[(env, exp)]]['FP'][0]
        FPiou = houses_detr.loc[[(env, exp)]]['FPiou'][0]


        table +=  ' & ' + str(TP) + ' (' + str(int(TP / total_positives *100)) + '\\%) & ' + str(FP) + ' (' + str(int(FP / total_positives * 100)) + '\\%) & ' + str(FPiou) + ' (' + str(int(FPiou / total_positives * 100)) + '\\%) '
        table += '\\\\ \n'
    table += '[2pt]\\hline \n'

print(table)
