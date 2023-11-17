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
datasets = ['igibson', 'deep_doors_2', 'gibson', 'gibson_deep_doors_2']
datasets_name = ['IGibson', 'DD2~\cite{deepdoors2}', 'Gibson', 'Gibson + DD2~\cite{deepdoors2}']
houses = ['floor1', 'floor4', 'chemistry_floor0', 'house_matteo']
#print(houses_detr.mean().index.tolist())




# Plots mean AP
colors = ['#2CA02C', '#FF7F0E',  '#D62728']
fig, ax = subplots(figsize=(10, 5))
dataframes = [houses_detr, houses_yolo, houses_faster]

X = np.arange(4)
#ax.bar(X, [0 for _ in range(3)], width=0.16)

for i, (dataframe, color) in enumerate(zip(dataframes, colors)):

    ax.bar(X + i * 0.2 + 0.04, [dataframe.loc[(dataframe['dataset'] == 'gibson_deep_doors_2') & (dataframe["label"] == 0) & (dataframe['house'] == house) & (dataframe['detector'] == 'GD')]['AP'].iloc[0] / 2 for house in houses],
               width=0.16,  color=color, edgecolor='#000000',alpha=0.9, hatch='/',
               linewidth=2)
    ax.bar(X + i * 0.2 + 0.04, [dataframe.loc[(dataframe['dataset'] == 'gibson_deep_doors_2') & (dataframe["label"] == 1) & (dataframe['house'] == house) & (dataframe['detector'] == 'GD')]['AP'].iloc[0] / 2 for house in houses],
               bottom=[dataframe.loc[(dataframe['dataset'] == 'gibson_deep_doors_2') & (dataframe["label"] == 0) & (dataframe['house'] == house) & (dataframe['detector'] == 'GD')]['AP'].iloc[0] / 2 for house in houses],
               width=0.16, color=color, edgecolor='#000000', alpha=0.9,
               linewidth=2)



ax.set_title(f'mAP of the general detectors', fontsize=18)

ax.set_ylim([0, 100])


matplotlib.pyplot.tick_params(left=True)
ax.tick_params(axis='y', labelsize=16)
ax.set_ylabel('mAP', fontsize=17)


matplotlib.pyplot.tick_params(bottom=True)
ax.set_xticks([i+0.34 for i in range(4)])
ax.set_xticklabels([f'$e_{i}$' for i in range(4)], fontsize=17)
ax.set_xlabel('Environment', fontsize=17)

ax.legend(prop={"size": 16}, bbox_to_anchor=(0.5, 0.95), loc='upper center', ncol=3, alignment='left')

fig.tight_layout()

def tikzplotlib_fix_ncols(obj):
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)
tikzplotlib_fix_ncols(fig)
chart_code = tikzplotlib.get_tikz_code().replace('\\begin{tikzpicture}', '\\begin{tikzpicture}[scale=0.725]')
chart_code = chart_code.replace('\\begin{axis}[', '\\begin{axis}[\nwidth=12cm,\nheight=7cm,')
chart_code = chart_code.replace('legend style={\n', 'legend cell align={left},\nlegend style={\n/tikz/every even column/.append style={column sep=0.3cm},\n')
chart_code = chart_code.replace('ybar legend', 'area legend')
chart_code = chart_code.replace('\\end{axis}', '\\input{graphics/legend_comparison_map}\n\\end{axis}')
text_file = open(f"../latex_plots/detector_comparison_general_detectors_ap.tex", "w")

#write string to file
text_file.write(chart_code)

#close file
text_file.close()

#tikzplotlib.save(f"../latex_plots/general_detector_{label}_e{env_number}.tex", axis_height='7cm', axis_width='12cm')
plt.show()



detectors = ['GD', 'QD_15', 'QD_25', 'QD_50', 'QD_75']
detectors_labels = ['$GD$', '$QD_{e}^{15}$', '$QD_{e}^{25}$', '$QD_{e}^{50}$', '$QD_{e}^{75}$']
fig, ax = subplots(figsize=(10, 5))
dataframes = [houses_detr, houses_yolo, houses_faster]

X = np.arange(5)
#ax.bar(X, [0 for _ in range(3)], width=0.16)

for i, (dataframe, color) in enumerate(zip(dataframes, colors)):

    ax.bar(X + i * 0.25 + 0.02, [dataframe.loc[(dataframe['dataset'] == 'gibson_deep_doors_2') & (dataframe["label"] == 0) & (dataframe['detector'] == detector)]['AP'].mean() / 2 for detector in detectors],
           width=0.2,  color=color, edgecolor='#000000',alpha=0.9, hatch='/',
           linewidth=2)

    mAP = dataframe.groupby(['house', 'detector', 'dataset'], as_index=False).mean(numeric_only=False)

    ax.bar(X + i * 0.25 + 0.02, [dataframe.loc[(dataframe['dataset'] == 'gibson_deep_doors_2') & (dataframe["label"] == 1) & (dataframe['detector'] == detector)]['AP'].mean() / 2 for detector in detectors],
           bottom=[dataframe.loc[(dataframe['dataset'] == 'gibson_deep_doors_2') & (dataframe["label"] == 0) & (dataframe['detector'] == detector)]['AP'].mean() / 2 for detector in detectors],
           yerr=[mAP.loc[(mAP['detector'] == detector) & (mAP['dataset'] == 'gibson_deep_doors_2')]['AP'].std() for detector in detectors],
           width=0.2, color=color, edgecolor='#000000', alpha=0.9,
           linewidth=2, capsize=3)



ax.set_title(f'mAP of the three detectors in real worlds', fontsize=18)

ax.set_ylim([0, 120])


matplotlib.pyplot.tick_params(left=True)
ax.tick_params(axis='y', labelsize=16)
ax.set_yticks([i * 20 for i in range(6)])
ax.set_yticklabels([f'{i * 20}' for i in range(6)], fontsize=17)
ax.set_ylabel('mAP', fontsize=17)


matplotlib.pyplot.tick_params(bottom=True)
ax.set_xticks([i + 0.27 for i in range(5)])
ax.set_xticklabels(detectors_labels, fontsize=17)
ax.set_xlabel('Environment', fontsize=17)

ax.legend(prop={"size": 16}, bbox_to_anchor=(0.5, 0.95), loc='upper center', ncol=5, alignment='left')

fig.tight_layout()

def tikzplotlib_fix_ncols(obj):
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)
tikzplotlib_fix_ncols(fig)
chart_code = tikzplotlib.get_tikz_code().replace('\\begin{tikzpicture}', '\\begin{tikzpicture}[scale=\\scale]')
chart_code = chart_code.replace('\\begin{axis}[', '\\begin{axis}[\nwidth=1.03\\textwidth,\nheight=0.5\\textwidth,')
chart_code = chart_code.replace('legend style={\n', 'legend cell align={left},\nlegend style={\n/tikz/every even column/.append style={column sep=0.3cm},\n')
chart_code = chart_code.replace('ybar legend', 'area legend')
chart_code = chart_code.replace('\\end{axis}', '\\input{graphics/legend_comparison_map}\n\\end{axis}')
text_file = open(f"../latex_plots/detector_comparison_map.tex", "w")

#write string to file
text_file.write(chart_code)

#close file
text_file.close()

#tikzplotlib.save(f"../latex_plots/general_detector_{label}_e{env_number}.tex", axis_height='7cm', axis_width='12cm')
plt.show()