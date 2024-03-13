import numpy as np
import pandas as pd

boxes = 100
quantity = 0.75
houses = ['floor1', 'floor4', 'chemistry_floor0', 'house_matteo']
metric_ap = pd.read_csv('../../../../results/filternet_results_ap_ablation.csv')
metric_complete = pd.read_csv('../../../../results/filternet_results_complete_ablation.csv')

metric_ap = metric_ap[(metric_ap['boxes'] == boxes) & (metric_ap['quantity'] == quantity)]
metric_complete = metric_complete[(metric_complete['boxes'] == boxes)& (metric_complete['quantity'] == quantity)]

metric_ap = metric_ap.groupby(['model', 'house', 'quantity', 'boxes', 'iou_threshold_matching',
                                           'confidence_threshold_tasknet', 'iou_threshold_tasknet',
                                           'confidence_threshold_filternet', 'iou_threshold_filternet',
                                            'relabeling', 'rescoring', 'suppression'], as_index=False).sum()


metric_complete = metric_complete.groupby(['model', 'house', 'quantity', 'boxes', 'iou_threshold_matching',
                          'confidence_threshold_tasknet', 'iou_threshold_tasknet',
                          'confidence_threshold_filternet', 'iou_threshold_filternet',
                                           'relabeling', 'rescoring', 'suppression'], as_index=False).sum()

metric_complete['TP_p'] = metric_complete['TP'] / metric_complete['total_positives']
metric_complete['FP_p'] = metric_complete['FP'] / metric_complete['total_positives']
metric_complete['FPiou_p'] = metric_complete['FPiou'] / metric_complete['total_positives']

table = ''
for i, (sup, res, rel) in enumerate([(r1, r2, s) for r1 in range(2) for r2 in range(2) for s in range(2)]):
    #if len([e for e in (rel, res, sup) if e == 1]) == 2:
        #continue
    model = 'filternet'
    table += f" \\{'' if rel == 0 else 'Vmark'} & \\{'' if res == 0 else 'Vmark'} & \\{'' if sup == 0 else 'Vmark'} "
    APs =[]
    TPs = []
    FPs = []
    FPious = []
    for e in houses:
        APs.append(metric_ap.loc[(metric_ap["house"] == e) & (metric_ap["model"] == model) & (metric_ap["relabeling"] == rel)& (metric_ap["rescoring"] == res)& (metric_ap["suppression"] == sup), "AP"].tolist()[0]/2*100)
        TPs.append(metric_complete.loc[(metric_complete["house"] == e) & (metric_complete["model"] == model) & (metric_complete["relabeling"] == rel)& (metric_complete["rescoring"] == res)& (metric_complete["suppression"] == sup), "TP_p"].tolist()[0]*100)
        FPs.append(metric_complete.loc[(metric_complete["house"] == e) & (metric_complete["model"] == model) & (metric_complete["relabeling"] == rel)& (metric_complete["rescoring"] == res)& (metric_complete["suppression"] == sup), "FP_p"].tolist()[0]*100)
        FPious.append(metric_complete.loc[(metric_complete["house"] == e) & (metric_complete["model"] == model) & (metric_complete["relabeling"] == rel)& (metric_complete["rescoring"] == res)& (metric_complete["suppression"] == sup), "FPiou_p"].tolist()[0]*100)

    table += (f'&{int(round(np.array(APs).mean(), 0))} &'
                  f'{int(round(np.array(TPs).mean(), 0))}\\% & '
                  f'{int(round(np.array(FPs).mean(), 0))}\\%  & '
                  f'{int(round(np.array(FPious).mean(), 0))}\\% ')
    table += '\\\\\n'
print(table)