import pandas as pd

houses = ['floor1', 'floor4', 'chemistry_floor0']

results = pd.read_excel('./../results/results_real_data_metric_complete.xlsx')

results = results[(results.epochs_gd == 60) & ((results.epochs_qd == 60) | (results.epochs_qd == 40))]

s = results.groupby('house').mean().std()

dataframes = []

for dataset in ['deep_doors_2', 'gibson', 'gibson_deep_doors_2']:
    for house in houses:
        d = results[(results.house == house) & (results.exp == 'GD') & (results.general_dataset == dataset)]
        for detector in ['QD_25', 'QD_50', 'QD_75']:
            d = d.append(results[(results.house == house) & (results.exp == detector) & (results.general_dataset == dataset)])
        dataframes.append(d)
        with pd.ExcelWriter('./../results/' + f'{house}_{dataset}_metric_complete.xlsx') as writer:
            d.to_excel(writer, sheet_name='s')

print(results)