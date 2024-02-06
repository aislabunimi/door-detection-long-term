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

houses = pd.read_excel('./../../../results/faster_rcnn_ap_real_data.xlsx')
houses['AP'] = houses['AP'].astype(np.float64)
houses['AP'] = houses['AP'].apply(lambda x: x*100).round()
houses = houses.loc[(houses['epochs_gd'] == 60) & ((houses['epochs_qd'] == 40) | (houses['epochs_qd'] == 60)) &
                    (houses['iou_threshold'] == iou_threshold) & (houses['confidence_threshold'] == confidence_threshold) &
                    (houses['detector'] == 'GD')]

houses_faster_ap = houses[['house', 'dataset', 'AP']].groupby(['house', 'dataset'], as_index=False).mean()

# YOLO
houses = pd.read_excel('./../../../results/yolov5_ap_real_data.xlsx')
houses['AP'] = houses['AP'].astype(np.float64)
houses['AP'] = houses['AP'].apply(lambda x: x*100).round()
houses = houses.loc[(houses['epochs_gd'] == 60) & ((houses['epochs_qd'] == 40) | (houses['epochs_qd'] == 60)) &
                    (houses['iou_threshold'] == iou_threshold) & (houses['confidence_threshold'] == confidence_threshold) & (houses['detector'] == 'GD')]

houses_yolo_ap = houses[['house', 'dataset', 'AP']].groupby(['house', 'dataset'], as_index=False).mean()

# DETR
houses = pd.read_excel('./../../../results/detr_ap_real_data.xlsx')
houses['AP'] = houses['AP'].astype(np.float64)
houses['AP'] = houses['AP'].apply(lambda x: x*100).round()
houses = houses.loc[(houses['epochs_gd'] == 60) & ((houses['epochs_qd'] == 40) | (houses['epochs_qd'] == 60)) &
                    (houses['iou_threshold'] == iou_threshold) & (houses['confidence_threshold'] == confidence_threshold) & (houses['detector'] == 'GD')]


houses['dataset'] = houses['dataset'].str.lower()
houses_detr_ap = houses[['house', 'dataset', 'AP']].groupby(['house', 'dataset'], as_index=False).mean()
houses_detr_ap.loc[houses_detr_ap['house'] == 'chemistryfloor0', 'house'] = 'chemistry_floor0'
houses_detr_ap.loc[houses_detr_ap['house'] == 'housematteo', 'house'] = 'house_matteo'


# Extended metric

houses = pd.read_excel('./../../../results/faster_rcnn_complete_metric_real_data.xlsx')
houses = houses.loc[(houses['epochs_gd'] == 60) & ((houses['epochs_qd'] == 40) | (houses['epochs_qd'] == 60)) &
                    (houses['iou_threshold'] == iou_threshold) & (houses['confidence_threshold'] == confidence_threshold) &
                    (houses['detector'] == 'GD')]

houses= houses.groupby(['iou_threshold', 'confidence_threshold', 'house', 'detector', 'dataset', 'epochs_gd', 'epochs_qd'], as_index=False).sum()
houses['TP_p'] = ((houses['TP'] / houses['total_positives']) * 100).round()
houses['FP_p'] = ((houses['FP'] / houses['total_positives']) * 100).round()
houses['FPiou_p'] = ((houses['FPiou'] / houses['total_positives']) * 100).round()
houses_faster = houses[['house', 'dataset', 'TP_p', 'FP_p', 'FPiou_p']].groupby(['house', 'dataset'], as_index=False).sum()
houses_faster.loc[houses_faster['house'] == 'chemistryfloor0', 'house'] = 'chemistry_floor0'
houses_faster.loc[houses_faster['house'] == 'housematteo', 'house'] = 'house_matteo'


# YOLO
houses = pd.read_excel('./../../../results/yolov5_complete_metric_real_data.xlsx')

houses = houses.loc[(houses['epochs_gd'] == 60) & ((houses['epochs_qd'] == 40) | (houses['epochs_qd'] == 60)) &
                    (houses['iou_threshold'] == iou_threshold) & (houses['confidence_threshold'] == confidence_threshold) & (houses['detector'] == 'GD')]

houses= houses.groupby(['iou_threshold', 'confidence_threshold', 'house', 'detector', 'dataset', 'epochs_gd', 'epochs_qd'], as_index=False).sum()
houses['TP_p'] = ((houses['TP'] / houses['total_positives']) * 100).round()
houses['FP_p'] = ((houses['FP'] / houses['total_positives']) * 100).round()
houses['FPiou_p'] = ((houses['FPiou'] / houses['total_positives']) * 100).round()
houses_yolo = houses[['house', 'dataset', 'TP_p', 'FP_p', 'FPiou_p']].groupby(['house', 'dataset'], as_index=False).sum()
houses_yolo.loc[houses_yolo['house'] == 'chemistryfloor0', 'house'] = 'chemistry_floor0'
houses_yolo.loc[houses_yolo['house'] == 'housematteo', 'house'] = 'house_matteo'


# DETR
houses = pd.read_excel('./../../../results/detr_complete_metrics_real_data.xlsx')

houses = houses.loc[(houses['epochs_gd'] == 60) & ((houses['epochs_qd'] == 40) | (houses['epochs_qd'] == 60)) &
                    (houses['iou_threshold'] == iou_threshold) & (houses['confidence_threshold'] == confidence_threshold) & (houses['detector'] == 'GD')]


houses['dataset'] = houses['dataset'].str.lower()
houses= houses.groupby(['iou_threshold', 'confidence_threshold', 'house', 'detector', 'dataset', 'epochs_gd', 'epochs_qd'], as_index=False).sum()
houses['TP_p'] = ((houses['TP'] / houses['total_positives']) * 100).round()
houses['FP_p'] = ((houses['FP'] / houses['total_positives']) * 100).round()
houses['FPiou_p'] = ((houses['FPiou'] / houses['total_positives']) * 100).round()
houses_detr = houses[['house', 'dataset', 'TP_p', 'FP_p', 'FPiou_p']].groupby(['house', 'dataset'], as_index=False).sum()
houses_detr.loc[houses_detr['house'] == 'chemistryfloor0', 'house'] = 'chemistry_floor0'
houses_detr.loc[houses_detr['house'] == 'housematteo', 'house'] = 'house_matteo'

model_names = ['DETR~\cite{detr}', 'YOLOv5~\cite{yolov5}', 'Faster~R--CNN~\cite{fasterrcnn}']
datasets = ['igibson', 'deep_doors_2', 'gibson', 'gibson_deep_doors_2']
datasets_name = ['\\DiG', '\\DDDtwo', '\\DG', '\\DDDtwoG']

table=''
for env_count, env in enumerate(['floor1', 'floor4', 'chemistry_floor0', 'house_matteo']):
    indexes = []
    for dataset_count, (dataset, dataset_label) in enumerate(zip(datasets, datasets_name)):
        d = []
        for dataframe_ap, dataframe_extended in zip([houses_detr_ap, houses_yolo_ap, houses_faster_ap], [houses_detr, houses_yolo, houses_faster]):
            d += ([
                round(dataframe_ap.loc[(dataframe_ap["house"] == env) & (dataframe_ap["dataset"] == dataset)]["AP"].iloc[0]),
                round(dataframe_extended.loc[(dataframe_extended["house"] == env) & (dataframe_extended["dataset"] == dataset)]["TP_p"].iloc[0]),
                round(dataframe_extended.loc[(dataframe_extended["house"] == env) & (dataframe_extended["dataset"] == dataset)]["FP_p"].iloc[0]),
                round(dataframe_extended.loc[(dataframe_extended["house"] == env) & (dataframe_extended["dataset"] == dataset)]["FPiou_p"].iloc[0]),
            ])
        indexes.append(d)
    indexes = np.array(indexes)
    indexes_str = indexes.astype(np.str_)
    for x_count, x in enumerate(indexes.T):
        firsts = np.argsort(x)
        if x_count % 4 == 0 or x_count %4 == 1:
            firsts = firsts[2:]
        else:
            firsts = firsts[:2][::-1]
        indexes_str[firsts[0]][x_count] = '\\underline{'+ indexes_str[firsts[0]][x_count] +'}'
        indexes_str[firsts[1]][x_count] = '\\textbf{'+ indexes_str[firsts[1]][x_count] +'}'

    for x_count, (x, dataset_label) in enumerate(zip(indexes_str, datasets_name)):
        table += '\multicolumn{1}{c}{\multirow{4}{*}{$e_'+str(env_count)+'$}} &' if x_count == 0 else '\multicolumn{1}{c}{} &'
        table += dataset_label + "& " + " & ".join(x)
        table+='\\\\\n'
    """
    for dataset_count, (dataset, dataset_label) in enumerate(zip(datasets, datasets_name)):
        table += '\multicolumn{1}{c}{\multirow{4}{*}{$e_'+str(env_count)+'$}} ' if dataset_count == 0 else '\multicolumn{1}{c}{} '
        table += f'& {dataset_label}'



        for dataframe_ap, dataframe_extended in zip([houses_detr_ap, houses_yolo_ap, houses_faster_ap], [houses_detr, houses_yolo, houses_faster]):

            table += (f'& {round(dataframe_ap.loc[(dataframe_ap["house"] == env) & (dataframe_ap["dataset"] == dataset)]["AP"].iloc[0])} & '
                      f'{round(dataframe_extended.loc[(dataframe_extended["house"] == env) & (dataframe_extended["dataset"] == dataset)]["TP_p"].iloc[0])} &'
                      f'{round(dataframe_extended.loc[(dataframe_extended["house"] == env) & (dataframe_extended["dataset"] == dataset)]["FP_p"].iloc[0])} &'
                      f'{round(dataframe_extended.loc[(dataframe_extended["house"] == env) & (dataframe_extended["dataset"] == dataset)]["FPiou_p"].iloc[0])}')
        table+='\\\\\n'
    """
    if env_count < 3:
        table+='[2pt]\hline\n'
print(table)