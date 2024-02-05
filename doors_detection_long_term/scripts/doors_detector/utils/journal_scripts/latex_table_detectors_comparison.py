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

# AP
houses = pd.read_excel('./../../../results/faster_rcnn_ap_real_data.xlsx')
houses['AP'] = houses['AP'].astype(np.float64)
houses['AP'] = houses['AP'].apply(lambda x: x*100).round()
houses = houses.loc[(houses['dataset'] != 'igibson') &(houses['dataset'] == 'gibson') & (houses['epochs_gd'] == 60) & ((houses['epochs_qd'] == 40) | (houses['epochs_qd'] == 60)) &
                    (houses['iou_threshold'] == iou_threshold) & (houses['confidence_threshold'] == confidence_threshold)]
houses = houses.groupby(['house', 'detector', 'dataset'], as_index=False).mean(numeric_only=False)
houses_faster_ap = houses

# YOLO
houses = pd.read_excel('./../../../results/yolov5_ap_real_data.xlsx')
houses['AP'] = houses['AP'].astype(np.float64)
houses['AP'] = houses['AP'].apply(lambda x: x*100).round()
houses = houses.loc[(houses['dataset'] != 'igibson') &(houses['dataset'] == 'gibson') & (houses['epochs_gd'] == 60) & ((houses['epochs_qd'] == 40) | (houses['epochs_qd'] == 60)) &
                    (houses['iou_threshold'] == iou_threshold) & (houses['confidence_threshold'] == confidence_threshold)]
houses = houses.groupby(['house', 'detector', 'dataset'], as_index=False).mean(numeric_only=False)
houses_yolo_ap = houses

# DETR
houses = pd.read_excel('./../../../results/detr_ap_real_data.xlsx')
houses['AP'] = houses['AP'].astype(np.float64)
houses['AP'] = houses['AP'].apply(lambda x: x*100).round()
houses = houses.loc[(houses['dataset'] != 'igibson') &(houses['dataset'] == 'GIBSON') & (houses['epochs_gd'] == 60) & ((houses['epochs_qd'] == 40) | (houses['epochs_qd'] == 60)) &
                    (houses['iou_threshold'] == iou_threshold) & (houses['confidence_threshold'] == confidence_threshold)]

houses = houses.groupby(['house', 'detector', 'dataset'], as_index=False).mean(numeric_only=False)
houses['dataset'] = houses['dataset'].str.lower()
houses_detr_ap = houses
houses_detr_ap.loc[houses_detr_ap['house'] == 'chemistryfloor0', 'house'] = 'chemistry_floor0'
houses_detr_ap.loc[houses_detr_ap['house'] == 'housematteo', 'house'] = 'house_matteo'

# Extended metric


houses = pd.read_excel('./../../../results/faster_rcnn_complete_metric_real_data.xlsx')
houses = houses.loc[(houses['dataset'] != 'igibson') &(houses['dataset'] == 'gibson') & (houses['epochs_gd'] == 60) & ((houses['epochs_qd'] == 40) | (houses['epochs_qd'] == 60)) &
                    (houses['iou_threshold'] == iou_threshold) & (houses['confidence_threshold'] == confidence_threshold)]

houses= houses.groupby(['iou_threshold', 'confidence_threshold', 'house', 'detector', 'dataset', 'epochs_gd', 'epochs_qd'], as_index=False).sum()
houses['TP_p'] = ((houses['TP'] / houses['total_positives']) * 100).round()
houses['FP_p'] = ((houses['FP'] / houses['total_positives']) * 100).round()
houses['FPiou_p'] = ((houses['FPiou'] / houses['total_positives']) * 100).round()
houses_faster = houses
houses_faster.loc[houses_faster['house'] == 'chemistryfloor0', 'house'] = 'chemistry_floor0'
houses_faster.loc[houses_faster['house'] == 'housematteo', 'house'] = 'house_matteo'


# YOLO
houses = pd.read_excel('./../../../results/yolov5_complete_metric_real_data.xlsx')

houses = houses.loc[(houses['dataset'] != 'igibson') &(houses['dataset'] == 'gibson') &  (houses['epochs_gd'] == 60) & ((houses['epochs_qd'] == 40) | (houses['epochs_qd'] == 60)) &
                    (houses['iou_threshold'] == iou_threshold) & (houses['confidence_threshold'] == confidence_threshold)]

houses= houses.groupby(['iou_threshold', 'confidence_threshold', 'house', 'detector', 'dataset', 'epochs_gd', 'epochs_qd'], as_index=False).sum()
houses['TP_p'] = ((houses['TP'] / houses['total_positives']) * 100).round()
houses['FP_p'] = ((houses['FP'] / houses['total_positives']) * 100).round()
houses['FPiou_p'] = ((houses['FPiou'] / houses['total_positives']) * 100).round()
houses_yolo = houses
houses_yolo.loc[houses_yolo['house'] == 'chemistryfloor0', 'house'] = 'chemistry_floor0'
houses_yolo.loc[houses_yolo['house'] == 'housematteo', 'house'] = 'house_matteo'


# DETR
houses = pd.read_excel('./../../../results/detr_complete_metrics_real_data.xlsx')

houses = houses.loc[(houses['dataset'] != 'igibson') &(houses['dataset'] == 'GIBSON') & (houses['epochs_gd'] == 60) & ((houses['epochs_qd'] == 40) | (houses['epochs_qd'] == 60)) &
                    (houses['iou_threshold'] == iou_threshold) & (houses['confidence_threshold'] == confidence_threshold)]


houses['dataset'] = houses['dataset'].str.lower()
houses= houses.groupby(['iou_threshold', 'confidence_threshold', 'house', 'detector', 'dataset', 'epochs_gd', 'epochs_qd'], as_index=False).sum()
houses['TP_p'] = ((houses['TP'] / houses['total_positives']) * 100).round()
houses['FP_p'] = ((houses['FP'] / houses['total_positives']) * 100).round()
houses['FPiou_p'] = ((houses['FPiou'] / houses['total_positives']) * 100).round()
houses_detr = houses
houses_detr.loc[houses_detr['house'] == 'chemistryfloor0', 'house'] = 'chemistry_floor0'
houses_detr.loc[houses_detr['house'] == 'housematteo', 'house'] = 'house_matteo'

model_names = ['DETR~\cite{detr}', 'YOLOv5~\cite{yolov5}', 'Faster~R--CNN~\cite{fasterrcnn}']
datasets = ['igibson', 'deep_doors_2', 'gibson', 'gibson_deep_doors_2']
datasets_name = ['IGibson', 'DD2~\cite{deepdoors2}', 'Gibson', 'Gibson + DD2~\cite{deepdoors2}']



model_names = ['DETR~\cite{detr}', 'YOLOv5~\cite{yolo}', 'Faster~R--CNN~\cite{fasterrcnn}']
datasets = ['igibson', 'deep_doors_2', 'gibson', 'gibson_deep_doors_2']
datasets_name = ['IGibson', 'DD2~\cite{deepdoors2}', 'Gibson', 'Gibson + DD2~\cite{deepdoors2}']
houses = ['floor1', 'floor4', 'chemistry_floor0', 'house_matteo']

# Plots mean AP
colors = ['#1F77B4','#2CA02C', '#FF7F0E', '#D62728', '#8C564B']
detectors = ['GD', 'QD_15', 'QD_25', 'QD_50', 'QD_75']
detectors_labels = ['$GD$', '$QD_{e}^{15}$', '$QD_{e}^{25}$', '$QD_{e}^{50}$', '$QD_{e}^{75}$']

data_mean = []
data_std = []

for detector in detectors:
    line_ap = []
    line_std = []

    for dataframe_ap, dataframe_extended in [(houses_detr_ap, houses_detr), (houses_yolo_ap, houses_yolo), (houses_faster_ap, houses_faster)]:
        mean = dataframe_ap.loc[(dataframe_ap['detector'] == detector)]['AP'].mean()
        std = dataframe_ap.loc[(dataframe_ap['detector'] == detector)]['AP'].std()
        line_ap += [mean]
        line_std += [std]

        mean = dataframe_extended.loc[(dataframe_extended['detector'] == detector)]['TP_p'].mean()
        std = dataframe_extended.loc[(dataframe_extended['detector'] == detector)]['TP_p'].std()
        line_ap += [mean]
        line_std += [std]

        mean = dataframe_extended.loc[(dataframe_extended['detector'] == detector)]['FP_p'].mean()
        std = dataframe_extended.loc[(dataframe_extended['detector'] == detector)]['FP_p'].std()
        line_ap += [mean]
        line_std += [std]

        mean = dataframe_extended.loc[(dataframe_extended['detector'] == detector)]['FPiou_p'].mean()
        std = dataframe_extended.loc[(dataframe_extended['detector'] == detector)]['FPiou_p'].std()
        line_ap += [mean]
        line_std += [std]
    data_mean.append(line_ap)
    data_std.append(line_std)

table = ''

for c, (means, stds) in enumerate(zip(data_mean, data_std)):
    table += detectors_labels[c]
    for i in range(4):

        v = [means[i], means[4+i], means[8+i]]
        if i < 2:
            firsts = np.argsort(v)[-2:]
        else:
            firsts = np.argsort(v)[:2][::-1]

        means[firsts[1]*4+i] = '\\textbf{' + str(int(round(means[firsts[1]*4+i],0))) +'}'

    for m, s in zip(means, stds):

        table += f'&${int(round(m,0)) if isinstance(m, float) else m} \\pm {int(round(s,0))} $'
    table += '\\\\[2pt]\n'
    if c%5 == 4:
        table+='\\hline\n'
print(table)