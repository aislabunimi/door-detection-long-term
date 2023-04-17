import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, font_manager
from matplotlib.pyplot import subplots
from pandas import CategoricalDtype

houses = pd.read_excel('./../../../results/detr_complete_metrics_real_data_different_conditions.xlsx')
houses = houses.loc[(houses['epochs_gd'] == 60) & ((houses['epochs_qd'] == 40) | (houses['epochs_qd'] == 60))]

labels = ['$GD_{-e}$', '$QD^{15}_e$', '$QD^{25}_e$', '$QD^{50}_e$', '$QD^{75}_e$']
experiments = ['GD', 'QD_15', 'QD_25', 'QD_50', 'QD_75']
datasets = ['GIBSON', 'DEEP_DOORS_2', 'GIBSON_DEEP_DOORS_2']
houses_list = ['floor1', 'floor4']
houses_list_dtype = CategoricalDtype(
    houses_list,
    ordered=True
)

a = houses.groupby(['house', 'detector', 'dataset', ]).sum()

print(a.loc[('floor1', 'GD', 'GIBSON')])

for dataset in datasets:
    print(f'Results for dataset {dataset}')
    for house in houses_list:
        print(f'\t{house}')
        for detector in experiments:
            line = a.loc[(house, detector, dataset)]
            print(f'\t\t{detector}: Total = {line["total_positives"]}, TP = {line["TP"]} ({round(line["TP"] / line["total_positives"]*100, 1)}%),'
                  f' FP = {line["FP"]} ({round(line["FP"] / line["total_positives"]*100, 1)}%), '
                  f'FPiou = {line["FPiou"]} ({round(line["FPiou"] / line["total_positives"]*100, 1)}%)')



