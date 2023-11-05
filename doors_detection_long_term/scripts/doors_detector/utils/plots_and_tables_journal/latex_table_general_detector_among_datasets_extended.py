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

houses_faster = houses

# YOLO
houses = pd.read_excel('./../../../results/yolov5_ap_real_data.xlsx')
houses['AP'] = houses['AP'].astype(np.float64)
houses['AP'] = houses['AP'].apply(lambda x: x*100).round()
houses = houses.loc[(houses['epochs_gd'] == 60) & ((houses['epochs_qd'] == 40) | (houses['epochs_qd'] == 60)) &
                    (houses['iou_threshold'] == iou_threshold) & (houses['confidence_threshold'] == confidence_threshold) & (houses['detector'] == 'GD')]

houses_yolo = houses

# DETR
houses = pd.read_excel('./../../../results/detr_ap_real_data.xlsx')
houses['AP'] = houses['AP'].astype(np.float64)
houses['AP'] = houses['AP'].apply(lambda x: x*100).round()
houses = houses.loc[(houses['epochs_gd'] == 60) & ((houses['epochs_qd'] == 40) | (houses['epochs_qd'] == 60)) &
                    (houses['iou_threshold'] == iou_threshold) & (houses['confidence_threshold'] == confidence_threshold) & (houses['detector'] == 'GD')]


houses['dataset'] = houses['dataset'].str.lower()
houses_detr = houses
houses_detr.loc[houses_detr['house'] == 'chemistryfloor0', 'house'] = 'chemistry_floor0'

model_names = ['DETR~\cite{detr}', 'YOLOv5~\cite{yolo}', 'Faster~R--CNN~\cite{fasterrcnn}']
datasets = ['deep_doors_2', 'gibson', 'gibson_deep_doors_2']
datasets_name = ['DD2~\cite{deepdoors2}', 'Gibson', 'Gibson + DD2~\cite{deepdoors2}']

#print(houses_detr.mean().index.tolist())




# Plots mean AP
colors = ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728']
for env_number, house in enumerate(['floor1', 'floor4', 'chemistry_floor0']):
    fig, ax = subplots(figsize=(10, 5))
    dataframes = [houses_detr, houses_yolo, houses_faster]

    X = np.arange(3)
    ax.bar(X, [0 for _ in range(3)], width=0.16)

    for i, (dataset, color) in enumerate(zip(datasets, colors[1:])):

        ax.bar(X + (i + 1) * 0.2 + 0.04, [d.loc[(d['dataset'] == dataset) & (d["label"] == 0) & (d['house'] == house)]['AP'].iloc[0] / 2 for d in dataframes],
               width=0.16,  color=color, edgecolor='#000000',alpha=0.9, hatch='/',
               linewidth=2)
        ax.bar(X + (i + 1) * 0.2 + 0.04, [d.loc[(d['dataset'] == dataset) & (d["label"] == 1) & (d['house'] == house)]['AP'].iloc[0] / 2 for d in dataframes],
               bottom=[d.loc[(d['dataset'] == dataset) & (d["label"] == 0) & (d['house'] == house)]['AP'].iloc[0] / 2 for d in dataframes],
               width=0.16, color=color, edgecolor='#000000', alpha=0.9,
               linewidth=2)



    ax.set_title(f'mAP results in $e_{env_number}$', fontsize=18)

    ax.set_ylim([0, 50])

    if env_number % 2 == 0:
        matplotlib.pyplot.tick_params(left=True)
        ax.tick_params(axis='y', labelsize=16)
        ax.set_ylabel('mAP', fontsize=17)
    else:
        matplotlib.pyplot.tick_params(left=False)

    if env_number<=1:
        matplotlib.pyplot.tick_params(bottom=False)
    if env_number >1:
        matplotlib.pyplot.tick_params(bottom=True)
        ax.set_xticks([i+0.34 for i in range(3)])
        ax.set_xticklabels(model_names, fontsize=17)
        ax.set_xlabel('Detector', fontsize=17)

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
    chart_code = chart_code.replace('\\end{axis}', '\\input{graphics/legend_ap_general_detector}\n\\end{axis}')
    text_file = open(f"../latex_plots/general_detector_stacked_mAP_e{env_number}.tex", "w")

    #write string to file
    text_file.write(chart_code)

    #close file
    text_file.close()

    #tikzplotlib.save(f"../latex_plots/general_detector_{label}_e{env_number}.tex", axis_height='7cm', axis_width='12cm')
    plt.show()