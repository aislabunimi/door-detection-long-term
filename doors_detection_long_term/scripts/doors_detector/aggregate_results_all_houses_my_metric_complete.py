import pandas as pd

houses = ['house_1', 'house_2', 'house_7', 'house_9', 'house_10', 'house_13', 'house_15', 'house_20', 'house_21', 'house_22']

results = pd.read_excel('./../results/epochs_and_backbone_analysis_my_metric_complete.xlsx')

results = results[(results.backbone == '2_layers') & (results.epochs_gd == 60) & ((results.epochs == 60) | (results.epochs == 40))]

s = results.groupby('house').mean().std()

dataframes = []

for house in houses:
    d = results[(results.house == house.replace('_', '')) & (results.detector == 'GD')]
    for detector in ['QD_25', 'QD_50', 'QD_75']:
        d = d.append(results[(results.house == house.replace('_', '')) & (results.detector == detector)])
    dataframes.append(d)
    with pd.ExcelWriter('./../results/' + f'{house}_metric_complete.xlsx') as writer:
        if not d.index.name:
            d.index.name = 'Index'
        d.to_excel(writer, sheet_name='s')

print(results)