import pandas as pd

boxes = 30
houses = ['floor1', 'floor4', 'chemistry_floor0', 'house_matteo']
metric_ap = pd.read_csv('../../../../results/filternet_results_ap.csv')
metric_complete = pd.read_csv('../../../../results/filternet_results_complete.csv')

metric_ap = metric_ap[metric_ap['boxes'] == boxes]
metric_complete = metric_complete[metric_complete['boxes'] == boxes]

metric_ap = metric_ap.groupby(['model', 'house', 'quantity', 'boxes', 'iou_threshold_matching',
                                           'confidence_threshold_tasknet', 'iou_threshold_tasknet',
                                           'confidence_threshold_filternet', 'iou_threshold_filternet'], as_index=False).sum()


metric_complete = metric_complete.groupby(['model', 'house', 'quantity', 'boxes', 'iou_threshold_matching',
                          'confidence_threshold_tasknet', 'iou_threshold_tasknet',
                          'confidence_threshold_filternet', 'iou_threshold_filternet'], as_index=False).sum()

metric_complete['TP_p'] = metric_complete['TP'] / metric_complete['total_positives']
metric_complete['FP_p'] = metric_complete['FP'] / metric_complete['total_positives']
metric_complete['FPiou_p'] = metric_complete['FPiou'] / metric_complete['total_positives']

table = ''
for i, quantity in enumerate([0.25, 0.25, 0.5, 0.75]):
    model = 'tasknet' if i == 0 else 'filternet'
    table += ' e '
    for e in houses:
        table += (f'&{int(round(metric_ap.loc[(metric_ap["house"] == e) & (metric_ap["model"] == model) & (metric_ap["quantity"] == quantity), "AP"].tolist()[0]/2*100, 0))} &'
                  f'{int(round(metric_complete.loc[(metric_complete["house"] == e) & (metric_complete["model"] == model) & (metric_complete["quantity"] == quantity), "TP_p"].tolist()[0]*100, 0))}\\% & '
                  f'{int(round(metric_complete.loc[(metric_complete["house"] == e) & (metric_complete["model"] == model) & (metric_complete["quantity"] == quantity), "FP_p"].tolist()[0]*100, 0))}\\%  & '
                  f'{int(round(metric_complete.loc[(metric_complete["house"] == e) & (metric_complete["model"] == model) & (metric_complete["quantity"] == quantity), "FPiou_p"].tolist()[0]*100, 0))}\\% ')
    table += '\\\\\n'
print(table)