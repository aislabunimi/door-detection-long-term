import numpy as np
import pandas
import pandas as pd
from pandas import CategoricalDtype

iou_threshold = 0.5
confidence_threshold = 0.75

houses = pd.read_excel('./../../results/faster_rcnn_ap_real_data.xlsx')
houses['AP'] = houses['AP'].astype(np.float64)
houses['AP'] = houses['AP'].apply(lambda x: x*100).round()
houses = houses.loc[(houses['epochs_gd'] == 60) & ((houses['epochs_qd'] == 40) | (houses['epochs_qd'] == 60)) &
                    (houses['iou_threshold'] == iou_threshold) & (houses['confidence_threshold'] == confidence_threshold) &
                    (houses['detector'] == 'GD')]


houses_faster = houses[['dataset', 'label', 'AP']]
houses_faster = houses_faster.groupby(['dataset', 'label', ])

# YOLO
houses = pd.read_excel('./../../results/yolov5_ap_real_data.xlsx')
houses['AP'] = houses['AP'].astype(np.float64)
houses['AP'] = houses['AP'].apply(lambda x: x*100).round()
houses = houses.loc[(houses['epochs_gd'] == 60) & ((houses['epochs_qd'] == 40) | (houses['epochs_qd'] == 60)) &
                    (houses['iou_threshold'] == iou_threshold) & (houses['confidence_threshold'] == confidence_threshold) & (houses['detector'] == 'GD')]

houses_yolo = houses[['dataset', 'label', 'AP']]
houses_yolo = houses_yolo.groupby(['dataset', 'label', ])


# DETR
houses = pd.read_excel('./../../results/detr_ap_real_data.xlsx')
houses['AP'] = houses['AP'].astype(np.float64)
houses['AP'] = houses['AP'].apply(lambda x: x*100).round()
houses = houses.loc[(houses['epochs_gd'] == 60) & ((houses['epochs_qd'] == 40) | (houses['epochs_qd'] == 60)) &
                    (houses['iou_threshold'] == iou_threshold) & (houses['confidence_threshold'] == confidence_threshold) & (houses['detector'] == 'GD')]


houses['dataset'] = houses['dataset'].str.lower()
houses_detr = houses[['dataset', 'label', 'AP']]
houses_detr = houses_detr.groupby(['dataset', 'label', ])

model_names = ['DETR~\cite{detr}', 'YOLOv5~\cite{yolo}', 'Faster~R--CNN~\cite{fasterrcnn}']
datasets = ['deep_doors_2', 'gibson', 'gibson_deep_doors_2']

print(houses_detr.mean().index.tolist())

table = ''
for model_name, mean, std in zip(model_names,
                        [houses_detr.mean(), houses_yolo.mean(), houses_faster.mean()],
                        [houses_detr.std(), houses_yolo.std(), houses_faster.std()]):
    table += '\multicolumn{1}{c|}{\multirow{2}{*}{' + model_name + '}} & '
    for c, (label_name, label) in enumerate(zip(['Closed', 'Open',], [0, 1])):
        if c != 0:
            table += '\multicolumn{1}{c|}{} & '

        table += label_name + ' & - & - '
        for i, dataset in enumerate(datasets):
            table += f'& {round(mean.loc[(dataset, label), "AP"])} & {round(std.loc[(dataset, label), "AP"])}'
        table += '\\\\ [2pt] \n'
    table += '\\hline \n'


print(table)