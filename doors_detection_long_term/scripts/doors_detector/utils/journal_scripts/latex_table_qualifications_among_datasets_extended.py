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
                    (houses['iou_threshold'] == iou_threshold) & (houses['confidence_threshold'] == confidence_threshold)]

houses_faster = houses

# YOLO
houses = pd.read_excel('./../../../results/yolov5_ap_real_data.xlsx')
houses['AP'] = houses['AP'].astype(np.float64)
houses['AP'] = houses['AP'].apply(lambda x: x*100).round()
houses = houses.loc[(houses['epochs_gd'] == 60) & ((houses['epochs_qd'] == 40) | (houses['epochs_qd'] == 60)) &
                    (houses['iou_threshold'] == iou_threshold) & (houses['confidence_threshold'] == confidence_threshold)]

houses_yolo = houses

# DETR
houses = pd.read_excel('./../../../results/detr_ap_real_data.xlsx')
houses['AP'] = houses['AP'].astype(np.float64)
houses['AP'] = houses['AP'].apply(lambda x: x*100).round()
houses = houses.loc[(houses['epochs_gd'] == 60) & ((houses['epochs_qd'] == 40) | (houses['epochs_qd'] == 60)) &
                    (houses['iou_threshold'] == iou_threshold) & (houses['confidence_threshold'] == confidence_threshold)]


houses['dataset'] = houses['dataset'].str.lower()
houses_detr = houses
houses_detr.loc[houses_detr['house'] == 'chemistryfloor0', 'house'] = 'chemistry_floor0'
houses_detr.loc[houses_detr['house'] == 'housematteo', 'house'] = 'house_matteo'

model_names = ['DETR~\cite{detr}', 'YOLOv5~\cite{yolo}', 'Faster~R--CNN~\cite{fasterrcnn}']
detectors = ['GD', 'QD_15', 'QD_25', 'QD_50', 'QD_75']
datasets_name = ['$GD$', '$QD_{e}^15$', '$QD_{e}^25$', '$QD_{e}^50$', '$QD_{e}^75$']

#print(houses_detr.mean().index.tolist())
houses_detr = houses_detr.loc[houses_detr['dataset'] != 'igibson']
houses_yolo = houses_yolo.loc[houses_yolo['dataset'] != 'igibson']
houses_faster = houses_faster.loc[houses_faster['dataset'] != 'igibson']

detectors = ['GD', 'QD_15', 'QD_25', 'QD_50', 'QD_75']
detectors_labels = ['$GD$', '$QD_{e}^{15}$', '$QD_{e}^{25}$', '$QD_{e}^{50}$', '$QD_{e}^{75}$']
performance = pd.concat([houses_detr.groupby(['house', 'detector', 'dataset'], as_index=False).mean(numeric_only=True),
                                houses_yolo.groupby(['house', 'detector', 'dataset'], as_index=False).mean(numeric_only=True),
                                houses_faster.groupby(['house', 'detector', 'dataset'], as_index=False).mean(numeric_only=True)], axis=0)
data = []

for house_number, house in enumerate(['floor1', 'floor4', 'chemistry_floor0', 'house_matteo']):
    for detector_count, (detector, detectors_label) in enumerate(zip(detectors, detectors_labels)):
        line = ['\multicolumn{1}{c}{\multirow{5}{*}{$e_{'+str(house_number)+'}$}}'] if detector_count == 0 else ['']
        line.append(detectors_label)
        for dataset in ['deep_doors_2', 'gibson', 'gibson_deep_doors_2']:
            mAPs = performance.loc[(performance['house'] == house) & (performance['dataset'] == dataset) & (performance['detector'] == detector)]['AP']
            mAP = round(mAPs.mean())
            mAPstd = round(mAPs.std())
            if detector_count == 0:
                line += [f'{mAP}', f'{mAPstd}','--', '--']
            else:
                incs = ((performance.loc[(performance['house'] == house) & (performance['dataset'] == dataset) & (performance['detector'] == detector)]['AP'].to_numpy()
                         - performance.loc[(performance['house'] == house) & (performance['dataset'] == dataset) & (performance['detector'] == detectors[detector_count-1])]['AP'].to_numpy()) *1.0
                    / performance.loc[(performance['house'] == house) & (performance['dataset'] == dataset) & (performance['detector'] == detectors[detector_count-1])]['AP'].to_numpy() *100)

                inc = round(incs.mean())
                incstd = round(incs.std())
                line += [f'{mAP}', f'{mAPstd}',f'{inc}\%', f'{incstd}']
        data.append(line)
print(data)

table = ''
for c, line in enumerate(data):
    v = [-1 for _ in range(11)]
    v[2] = int(line[2])
    v[6] = int(line[6])
    v[10] = int(line[10])
    firsts = np.argsort(v)[-2:]
    if v[firsts[0]] == v[firsts[1]] and firsts[0] < firsts[1]:
        firsts = firsts[::-1]
    line[firsts[0]] = '\\underline{' + line[firsts[0]] +'}'
    line[firsts[1]] = '\\textbf{' + line[firsts[1]] +'}'
    table+=(' & ').join(line) + '\\\\[2pt]\n'
    if c%5 == 4:
        table+='\\hline\n'

print(table)






