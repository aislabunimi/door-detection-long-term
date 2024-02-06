import matplotlib
import numpy as np
import pandas
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.pyplot import subplots
import tikzplotlib

iou_threshold = 0.5
confidence_threshold = 0.75


# Extended metric
houses = pd.read_excel('./../../../results/faster_rcnn_complete_metric_real_data_different_conditions.xlsx')
houses = houses.loc[(houses['epochs_gd'] == 60) & ((houses['epochs_qd'] == 40) | (houses['epochs_qd'] == 60)) &
                    (houses['iou_threshold'] == iou_threshold) & (houses['confidence_threshold'] == confidence_threshold) ]

houses = houses.groupby(['iou_threshold', 'confidence_threshold', 'house', 'detector', 'dataset', 'epochs_gd', 'epochs_qd'], as_index=False).sum()
houses['TP_p'] = ((houses['TP'] / houses['total_positives']) * 100).round()
houses['FP_p'] = ((houses['FP'] / houses['total_positives']) * 100).round()
houses['FPiou_p'] = ((houses['FPiou'] / houses['total_positives']) * 100).round()
houses_faster = houses[['house', 'dataset', 'detector', 'TP_p', 'FP_p', 'FPiou_p']].groupby(['house', 'dataset', 'detector'], as_index=False).sum()
houses_faster.loc[houses_faster['house'] == 'chemistryfloor0', 'house'] = 'chemistry_floor0'
houses_faster.loc[houses_faster['house'] == 'housematteo', 'house'] = 'house_matteo'

# YOLO
houses = pd.read_excel('./../../../results/yolov5_complete_metric_real_data_different_condition.xlsx')

houses = houses.loc[(houses['epochs_gd'] == 60) & ((houses['epochs_qd'] == 40) | (houses['epochs_qd'] == 60)) &
                    (houses['iou_threshold'] == iou_threshold) & (houses['confidence_threshold'] == confidence_threshold)]

houses= houses.groupby(['iou_threshold', 'confidence_threshold', 'house', 'detector', 'dataset', 'epochs_gd', 'epochs_qd'], as_index=False).sum()
houses['TP_p'] = ((houses['TP'] / houses['total_positives']) * 100).round()
houses['FP_p'] = ((houses['FP'] / houses['total_positives']) * 100).round()
houses['FPiou_p'] = ((houses['FPiou'] / houses['total_positives']) * 100).round()
houses_yolo = houses[['house', 'dataset', 'detector', 'TP_p', 'FP_p', 'FPiou_p']].groupby(['house', 'dataset', 'detector'], as_index=False).sum()
houses_yolo.loc[houses_yolo['house'] == 'chemistryfloor0', 'house'] = 'chemistry_floor0'
houses_yolo.loc[houses_yolo['house'] == 'housematteo', 'house'] = 'house_matteo'


# DETR
houses = pd.read_excel('./../../../results/detr_complete_metrics_real_data_different_conditions.xlsx')

houses = houses.loc[(houses['epochs_gd'] == 60) & ((houses['epochs_qd'] == 40) | (houses['epochs_qd'] == 60)) &
                    (houses['iou_threshold'] == iou_threshold) & (houses['confidence_threshold'] == confidence_threshold)]


houses['dataset'] = houses['dataset'].str.lower()
houses= houses.groupby(['iou_threshold', 'confidence_threshold', 'house', 'detector', 'dataset', 'epochs_gd', 'epochs_qd'], as_index=False).sum()
houses['TP_p'] = ((houses['TP'] / houses['total_positives']) * 100).round()
houses['FP_p'] = ((houses['FP'] / houses['total_positives']) * 100).round()
houses['FPiou_p'] = ((houses['FPiou'] / houses['total_positives']) * 100).round()
houses_detr = houses[['house', 'dataset', 'TP_p', 'FP_p', 'FPiou_p','detector']].groupby(['house', 'dataset', 'detector'], as_index=False).sum()
houses_detr.loc[houses_detr['house'] == 'chemistryfloor0', 'house'] = 'chemistry_floor0'
houses_detr.loc[houses_detr['house'] == 'housematteo', 'house'] = 'house_matteo'

houses_detr = houses_detr.loc[houses_detr['dataset'] != 'igibson']
houses_yolo = houses_yolo.loc[houses_yolo['dataset'] != 'igibson']
houses_faster = houses_faster.loc[houses_faster['dataset'] != 'igibson']
performance_extended = pd.concat([houses_detr.groupby(['house', 'detector', 'dataset'], as_index=False).mean(numeric_only=True),
                                  houses_yolo.groupby(['house', 'detector', 'dataset'], as_index=False).mean(numeric_only=True),
                                  houses_faster.groupby(['house', 'detector', 'dataset'], as_index=False).mean(numeric_only=True)], axis=0)

# mAP


houses = pd.read_excel('./../../../results/faster_rcnn_ap_real_data_different_conditions.xlsx')
houses['AP'] = houses['AP'].astype(np.float64)
houses['AP'] = houses['AP'].apply(lambda x: x*100).round()
houses = houses.loc[(houses['epochs_gd'] == 60) & ((houses['epochs_qd'] == 40) | (houses['epochs_qd'] == 60)) &
                    (houses['iou_threshold'] == iou_threshold) & (houses['confidence_threshold'] == confidence_threshold)]

houses_faster_ap = houses

# YOLO
houses = pd.read_excel('./../../../results/yolov5_ap_real_data_different_condition.xlsx')
houses['AP'] = houses['AP'].astype(np.float64)
houses['AP'] = houses['AP'].apply(lambda x: x*100).round()
houses = houses.loc[(houses['epochs_gd'] == 60) & ((houses['epochs_qd'] == 40) | (houses['epochs_qd'] == 60)) &
                    (houses['iou_threshold'] == iou_threshold) & (houses['confidence_threshold'] == confidence_threshold)]

houses_yolo_ap = houses

# DETR
houses = pd.read_excel('./../../../results/detr_ap_real_data_different_conditions.xlsx')
houses['AP'] = houses['AP'].astype(np.float64)
houses['AP'] = houses['AP'].apply(lambda x: x*100).round()
houses = houses.loc[(houses['epochs_gd'] == 60) & ((houses['epochs_qd'] == 40) | (houses['epochs_qd'] == 60)) &
                    (houses['iou_threshold'] == iou_threshold) & (houses['confidence_threshold'] == confidence_threshold)]


houses['dataset'] = houses['dataset'].str.lower()
houses_detr_ap = houses
houses_detr_ap.loc[houses_detr_ap['house'] == 'chemistryfloor0', 'house'] = 'chemistry_floor0'
houses_detr_ap.loc[houses_detr_ap['house'] == 'housematteo', 'house'] = 'house_matteo'

model_names = ['DETR~\cite{detr}', 'YOLOv5~\cite{yolo}', 'Faster~R--CNN~\cite{fasterrcnn}']
detectors = ['GD', 'QD_15', 'QD_25', 'QD_50', 'QD_75']
datasets_name = ['$GD$', '$QD_{e}^15$', '$QD_{e}^25$', '$QD_{e}^50$', '$QD_{e}^75$']

#print(houses_detr.mean().index.tolist())
houses_detr_ap = houses_detr_ap.loc[houses_detr_ap['dataset'] != 'igibson']
houses_yolo_ap = houses_yolo_ap.loc[houses_yolo_ap['dataset'] != 'igibson']
houses_faster_ap = houses_faster_ap.loc[houses_faster_ap['dataset'] != 'igibson']

performance_ap = pd.concat([houses_detr_ap.groupby(['house', 'detector', 'dataset'], as_index=False).mean(numeric_only=True),
                            houses_yolo_ap.groupby(['house', 'detector', 'dataset'], as_index=False).mean(numeric_only=True),
                            houses_faster_ap.groupby(['house', 'detector', 'dataset'], as_index=False).mean(numeric_only=True)], axis=0)



model_names = ['DETR~\cite{detr}', 'YOLOv5~\cite{yolov5}', 'Faster~R--CNN~\cite{fasterrcnn}']
datasets = ['igibson', 'deep_doors_2', 'gibson', 'gibson_deep_doors_2']
datasets_name = ['$\mathcal{D}_{ig}$', '$\mathcal{D}_{dd2}$', '$\mathcal{D}_{g}$', '$\mathcal{D}_{dd2+g}$']

detectors = ['GD', 'QD_15', 'QD_25', 'QD_50', 'QD_75']
detectors_labels = ['$GD$', '$QD_{e}^{15}$', '$QD_{e}^{25}$', '$QD_{e}^{50}$', '$QD_{e}^{75}$']

means = []
stds =[]

for house_number, house in enumerate(['floor1', 'floor4']):
    for detector_count, (detector, detectors_label) in enumerate(zip(detectors, detectors_labels)):
        line_map = []
        line_std = []
        for dataset in ['deep_doors_2', 'gibson', 'gibson_deep_doors_2']:
            mAPs = performance_ap.loc[(performance_ap['house'] == house) & (performance_ap['dataset'] == dataset) & (performance_ap['detector'] == detector)]['AP']
            mAP = mAPs.mean()
            mAPstd = mAPs.std()

            TP_s = performance_extended.loc[(performance_extended['house'] == house) & (performance_extended['dataset'] == dataset) & (performance_extended['detector'] == detector)]['TP_p']
            TP_mean = TP_s.mean()
            TP_std = TP_s.std()

            FP_s = performance_extended.loc[(performance_extended['house'] == house) & (performance_extended['dataset'] == dataset) & (performance_extended['detector'] == detector)]['FP_p']
            FP_mean = FP_s.mean()
            FP_std = FP_s.std()
            FPiou_s = performance_extended.loc[(performance_extended['house'] == house) & (performance_extended['dataset'] == dataset) & (performance_extended['detector'] == detector)]['FPiou_p']
            FPiou_mean = FPiou_s.mean()
            FPiou_std = FPiou_s.std()

            line_map += [mAP,TP_mean, FP_mean, FPiou_mean]
            line_std += [mAPstd, TP_std, FP_std, FPiou_std]
        means.append(line_map)
        stds.append(line_std)

table = ''
for c, (line_map, line_std) in enumerate(zip(means, stds)):
    table += '\multicolumn{1}{c}{\multirow{5}{*}{$e_'+str(c//5)+'$}} &' if c%5 == 0 else '\multicolumn{1}{c}{} &'
    table += '' + detectors_labels[c%5]
    for i in range(4):

        v = [line_map[i], line_map[4+i], line_map[8+i]]


        if i < 2:
            firsts = np.argsort(v)[-2:]

        else:
            firsts = np.argsort(v)[:2][::-1]


        line_map[firsts[1]*4+i] = '\\textbf{' + str(int(round(line_map[firsts[1]*4+i],0))) +'}'

    for m, s in zip(line_map, line_std):

        table += f'&${int(round(m,0)) if isinstance(m, float) else m} \\pm {int(round(s, 0))} $'
    table += '\\\\[2pt]\n'
    if c%5 == 4:
        table+='\\hline\n'

print(table)






