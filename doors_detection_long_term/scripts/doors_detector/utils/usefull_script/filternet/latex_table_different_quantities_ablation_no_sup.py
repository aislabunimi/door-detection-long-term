import numpy as np
import pandas as pd

boxes = 100
quantity = 0.75
houses = ['floor1', 'floor4', 'chemistry_floor0', 'house_matteo']
metric_ap = pd.read_csv('../../../../results/filternet_results_ap_no_sup.csv')
metric_complete = pd.read_csv('../../../../results/filternet_results_complete_no_sup.csv')

metric_ap = metric_ap[(metric_ap['boxes'] == boxes) & (metric_ap['quantity'] == quantity)]
metric_complete = metric_complete[(metric_complete['boxes'] == boxes)& (metric_complete['quantity'] == quantity)]

metric_ap = metric_ap.groupby(['model', 'house', 'quantity', 'boxes', 'iou_threshold_matching',
                                           'confidence_threshold_tasknet', 'iou_threshold_tasknet',
                                           'confidence_threshold_filternet', 'iou_threshold_filternet',
                                           ], as_index=False).sum()


metric_complete = metric_complete.groupby(['model', 'house', 'quantity', 'boxes', 'iou_threshold_matching',
                          'confidence_threshold_tasknet', 'iou_threshold_tasknet',
                          'confidence_threshold_filternet', 'iou_threshold_filternet',
                                           ], as_index=False).sum()

metric_complete['TP_p'] = metric_complete['TP'] / metric_complete['total_positives']
metric_complete['FP_p'] = metric_complete['FP'] / metric_complete['total_positives']
metric_complete['FPiou_p'] = metric_complete['FPiou'] / metric_complete['total_positives']

table = ''
model = 'filternet'
APs =[]
TPs = []
FPs = []
FPious = []
for e in houses:
    APs.append(metric_ap.loc[(metric_ap["house"] == e) & (metric_ap["model"] == model), "AP"].tolist()[0]/2*100)
    TPs.append(metric_complete.loc[(metric_complete["house"] == e) & (metric_complete["model"] == model) , "TP_p"].tolist()[0]*100)
    FPs.append(metric_complete.loc[(metric_complete["house"] == e) & (metric_complete["model"] == model) , "FP_p"].tolist()[0]*100)
    FPious.append(metric_complete.loc[(metric_complete["house"] == e) & (metric_complete["model"] == model), "FPiou_p"].tolist()[0]*100)

table += (f'&{int(round(np.array(APs).mean(), 0))} &'
              f'{int(round(np.array(TPs).mean(), 0))}\\% & '
              f'{int(round(np.array(FPs).mean(), 0))}\\%  & '
              f'{int(round(np.array(FPious).mean(), 0))}\\% ')
table += '\\\\\n'
print(table)