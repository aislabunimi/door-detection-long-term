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

houses = pd.read_excel('./../../../results/faster_rcnn_complete_metric_real_data.xlsx')
houses = houses.loc[(houses['epochs_gd'] == 60) & ((houses['epochs_qd'] == 40) | (houses['epochs_qd'] == 60)) &
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

houses = houses.loc[(houses['epochs_gd'] == 60) & ((houses['epochs_qd'] == 40) | (houses['epochs_qd'] == 60)) &
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

houses = houses.loc[(houses['epochs_gd'] == 60) & ((houses['epochs_qd'] == 40) | (houses['epochs_qd'] == 60)) &
                    (houses['iou_threshold'] == iou_threshold) & (houses['confidence_threshold'] == confidence_threshold)]


houses['dataset'] = houses['dataset'].str.lower()
houses= houses.groupby(['iou_threshold', 'confidence_threshold', 'house', 'detector', 'dataset', 'epochs_gd', 'epochs_qd'], as_index=False).sum()
houses['TP_p'] = ((houses['TP'] / houses['total_positives']) * 100).round()
houses['FP_p'] = ((houses['FP'] / houses['total_positives']) * 100).round()
houses['FPiou_p'] = ((houses['FPiou'] / houses['total_positives']) * 100).round()
houses_detr = houses
houses_detr.loc[houses_detr['house'] == 'chemistryfloor0', 'house'] = 'chemistry_floor0'
houses_detr.loc[houses_detr['house'] == 'housematteo', 'house'] = 'house_matteo'

model_names = ['DETR~\cite{detr}', 'YOLOv5~\cite{yolo}', 'Faster~R--CNN~\cite{fasterrcnn}']
datasets = ['igibson', 'deep_doors_2', 'gibson', 'gibson_deep_doors_2']
datasets_name = ['IGibson', 'DD2~\cite{deepdoors2}', 'Gibson', 'Gibson + DD2~\cite{deepdoors2}']

#print(houses_detr.mean().index.tolist())

houses_detr = houses_detr.loc[houses_detr['dataset'] != 'igibson']
houses_yolo = houses_yolo.loc[houses_yolo['dataset'] != 'igibson']
houses_faster = houses_faster.loc[houses_faster['dataset'] != 'igibson']


# Plots mean AP
colors = ['#1F77B4','#2CA02C', '#FF7F0E', '#D62728', '#8C564B']
detectors = ['GD', 'QD_15', 'QD_25', 'QD_50', 'QD_75']
detectors_labels = ['$GD$', '$QD_{e}^{15}$', '$QD_{e}^{25}$', '$QD_{e}^{50}$', '$QD_{e}^{75}$']

colors = ['#1F77B4','#2CA02C', '#FF7F0E', '#D62728', '#8C564B']
detectors = ['GD', 'QD_15', 'QD_25', 'QD_50', 'QD_75']


fig, ax = subplots(figsize=(10, 5))
dataframes = [houses_detr.loc[houses_detr['dataset'] == 'gibson_deep_doors_2'],
              houses_yolo.loc[houses_yolo['dataset'] == 'gibson_deep_doors_2'],
              houses_faster.loc[houses_faster['dataset'] == 'gibson_deep_doors_2']]

X = np.arange(3)
#ax.bar(X, [0 for _ in range(3)], width=0.16)

for data_count, data in enumerate(dataframes):
    means = np.array([data.loc[(data['detector'] == detector)]['TP_p'].mean() for detector in detectors])
    stds = np.array([data.loc[(data['detector'] == detector)]['TP_p'].std() for detector in detectors])
    ax.plot([i + 5*data_count for i in range(5)], means,
           color='#2CA02C', marker='o',markersize=7)
    ax.fill_between([i + 5*data_count for i in range(5)], means + stds, means - stds,
                    color='#2CA02C', alpha=.2)

    means = np.array([data.loc[(data['detector'] == detector)]['FP_p'].mean() for detector in detectors])
    stds = np.array([data.loc[(data['detector'] == detector)]['FP_p'].std() for detector in detectors])
    ax.plot([i+ 5*data_count for i in range(5)], means,
           color='#D62728', marker='^',markersize=7 )
    ax.fill_between([i+ 5*data_count for i in range(5)], means + stds, means - stds,
                    color='#D62728', alpha=.2)

    means = np.array([data.loc[(data['detector'] == detector)]['FPiou_p'].mean() for detector in detectors])
    stds = np.array([data.loc[(data['detector'] == detector)]['FPiou_p'].std() for detector in detectors])
    ax.plot([i+ 5*data_count for i in range(5)], means,
           color='#FF7F0E', marker='d',markersize=7)
    ax.fill_between([i+ 5*data_count for i in range(5)], means + stds, means - stds,
                    color='#FF7F0E', alpha=.2)


ax.set_title(f'Extended metric results in $', fontsize=18)
ax.axhline(y=0.0, linewidth=1, color='black')
ax.set_ylim([0, 115])


matplotlib.pyplot.tick_params(left=True)
ax.tick_params(axis='y', labelsize=16)
ax.set_yticks([20 * i for i in range(5)])
ax.set_yticklabels([f'{20 * i}' for i in range(5)])
ax.set_ylabel('%', fontsize=17)



matplotlib.pyplot.tick_params(bottom=True)
ax.set_xticks([i*5 + 2 for i in range(3)])

ax.set_xticklabels(model_names, fontsize=5)
ax.set_xlabel('Detector', fontsize=17)

labels = detectors_labels + detectors_labels + detectors_labels
for c, l in enumerate(labels):
    ax.text(c, 107, l, rotation=45, fontsize=15, horizontalalignment='center', verticalalignment='center')

ax.vlines(x = 4.5, ymin = 0, ymax = 115, color='gray', linestyle='--',alpha=0.7)
ax.vlines(x = 9.5, ymin = 0, ymax = 115, color='gray', linestyle='--', alpha=0.7)

ax.legend(prop={"size": 16}, bbox_to_anchor=(0.5, 0.97), loc='upper center', ncol=4, alignment='left')
ax.set_yticklabels([item.get_text().replace(chr(8722), '') for item in ax.get_yticklabels()])
fig.tight_layout()

def tikzplotlib_fix_ncols(obj):
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)
tikzplotlib_fix_ncols(fig)
chart_code = tikzplotlib.get_tikz_code().replace('\\begin{tikzpicture}', '\\begin{tikzpicture}[scale=\scale]')
chart_code = chart_code.replace('\\begin{axis}[', '\\begin{axis}[\nwidth=\\textwidth,\nheight=0.5\\textwidth,')
chart_code = chart_code.replace('legend style={\n', 'legend cell align={left},\nlegend style={\n/tikz/every even column/.append style={column sep=0.3cm},\n')
chart_code = chart_code.replace('ybar legend', 'area legend')
chart_code = chart_code.replace('\\end{axis}', '\\input{graphics/legend_comparison_extended_metric}\n\\end{axis}')
chart_code = chart_code.replace('mark size=3', 'mark size=2')
text_file = open(f"../latex_plots/qualified_detectors_comparison_extended_metric.tex", "w")

#write string to file
text_file.write(chart_code)

#close file
text_file.close()

#tikzplotlib.save(f"../latex_plots/general_detector_{label}_e{env_number}.tex", axis_height='7cm', axis_width='12cm')
plt.show()