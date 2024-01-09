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



# Plots qualifications over all datasets
colors = ['#1F77B4','#2CA02C', '#FF7F0E', '#D62728', '#8C564B']
#B0B0B0
#17B0CF
#8C564B
for env_number, house in enumerate(['floor1', 'floor4', 'chemistry_floor0', 'house_matteo']):
    fig, ax = subplots(figsize=(10, 5))
    dataframes = [houses_detr,
                  houses_yolo,
                  houses_faster]

    X = np.arange(3)
    #ax.bar(X, [0 for _ in range(3)], width=0.16)

    for i, (detector, color) in enumerate(zip(detectors, colors)):

        ax.bar(X + i * 0.16 + 0.02, [d.loc[(d['detector'] == detector) & (d["label"] == 0) & (d['house'] == house)]['AP'].mean() / 2 for d in dataframes],
               width=0.12, color=color, edgecolor='#000000',alpha=0.9, hatch='/',
               linewidth=2)
        ax.bar(X + i * 0.16 + 0.02, [d.loc[(d['detector'] == detector) & (d["label"] == 1) & (d['house'] == house)]['AP'].mean() / 2 for d in dataframes],
               bottom=[d.loc[(d['detector'] == detector) & (d["label"] == 0) & (d['house'] == house)]['AP'].mean() / 2 for d in dataframes],
               yerr=[d.loc[(d['detector'] == detector) & (d['house'] == house)].groupby(['dataset']).mean(numeric_only=True)['AP'].std() for d in dataframes],
               width=0.12, color=color, edgecolor='#000000', alpha=0.9,
               linewidth=2,capsize=3)



    ax.set_title(f'mAP results in $e_{env_number}$', fontsize=18)

    ax.set_ylim([0, 130])

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
    chart_code = chart_code.replace('\\end{axis}', '\\input{graphics/legend_ap_qualified_detector}\n\\end{axis}')
    text_file = open(f"../latex_plots/qualified_detector_stacked_mAP_e{env_number}.tex", "w")

    #write string to file
    text_file.write(chart_code)

    #close file
    text_file.close()

    #tikzplotlib.save(f"../latex_plots/general_detector_{label}_e{env_number}.tex", axis_height='7cm', axis_width='12cm')
    plt.show()


# Plots comparison between dataset
colors = ['#2CA02C', '#FF7F0E', '#D62728',]
datasets = ['deep_doors_2', 'gibson', 'gibson_deep_doors_2']
detectors_labels = ['$GD$', '$QD_{e}^{15}$', '$QD_{e}^{25}$', '$QD_{e}^{50}$', '$QD_{e}^{75}$']
for env_number, house in enumerate(['floor1', 'floor4', 'chemistry_floor0', 'house_matteo']):
    fig, ax = subplots(figsize=(10, 5))
    #ax.bar(X, [0 for _ in range(3)], width=0.16)
    performance = pd.concat([houses_detr.reset_index(), houses_yolo.reset_index(), houses_faster.reset_index()], axis=0)
    X = np.arange(5)
    for i, (dataset, color) in enumerate(zip(datasets, colors)):
        #ax.plot([i for i in range(5)], [performance.loc[(performance['dataset'] == dataset) & (performance['detector'] == d)]['AP'].mean() for d in detectors],
         #       color=color)
        ax.bar(X + i * 0.25 + 0.02, [performance.loc[(performance['house'] == house) & (performance['dataset'] == dataset) & (performance['detector'] == d) & (performance['label'] == 0)]['AP'].mean() / 2 for d in detectors],
               width=0.2, color=color, edgecolor='#000000', alpha=0.9, hatch='/',
               linewidth=2)
        mAPs = pd.concat([houses_detr.groupby(['house', 'detector', 'dataset'], as_index=False).mean(numeric_only=True),
                          houses_yolo.groupby(['house', 'detector', 'dataset'], as_index=False).mean(numeric_only=True),
                          houses_faster.groupby(['house', 'detector', 'dataset'], as_index=False).mean(numeric_only=True)], axis=0)
        ax.bar(X + i * 0.25 + 0.02, [performance.loc[(performance['house'] == house) & (performance['dataset'] == dataset) & (performance['detector'] == d) & (performance['label'] == 1)]['AP'].mean() / 2 for d in detectors],
               width=0.2, color=color, edgecolor='#000000',alpha=0.9,
               yerr=[mAPs.loc[(mAPs['house'] == house) & (mAPs['dataset'] == dataset) & (mAPs['detector'] == d)]['AP'].std() for d in detectors],
               bottom=[performance.loc[(performance['house'] == house) & (performance['dataset'] == dataset) & (performance['detector'] == d) & (performance['label'] == 0)]['AP'].mean() / 2 for d in detectors],
               linewidth=2,
               capsize=3)



    ax.set_title(f'mAP results over the detectors in $e_{env_number}$', fontsize=18)

    ax.set_ylim([0, 100])

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
        ax.set_xticks([i+0.27 for i in range(5)])
        ax.set_xticklabels(detectors_labels, fontsize=17)
        ax.set_xlabel('Qualification rounds', fontsize=17)

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
    #chart_code = chart_code.replace('\\end{axis}', '\\input{graphics/legend_ap_general_detector_over_dataset}\n\\end{axis}')
    text_file = open(f"../latex_plots/qualified_detector_over_datasets_mAP_d_{env_number}.tex", "w")

    #write string to file
    text_file.write(chart_code)

    #close file
    text_file.close()

    #tikzplotlib.save(f"../latex_plots/general_detector_{label}_e{env_number}.tex", axis_height='7cm', axis_width='12cm')
    plt.show()


# Plot label dataset

# Plots comparison between dataset
colors = ['#1F77B4','#2CA02C', '#FF7F0E', '#D62728', '#8C564B']
datasets = ['deep_doors_2', 'gibson', 'gibson_deep_doors_2']
dataframe = pd.concat([houses_detr.reset_index(), houses_yolo.reset_index(), houses_faster.reset_index()], axis=0)
for label_count, (label_name, label) in enumerate(zip(['Closed door', 'Open door'], [0, 1])):
    fig, ax = subplots(figsize=(10, 5))
    #ax.bar(X, [0 for _ in range(3)], width=0.16)
    performance = dataframe
    X = np.arange(5)
    for i, (dataset, color) in enumerate(zip(datasets, colors[:-1])):
        #ax.plot([i for i in range(5)], [performance.loc[(performance['dataset'] == dataset) & (performance['detector'] == d)]['AP'].mean() for d in detectors],
        #       color=color)
        ax.bar(X + i * 0.16 + 0.02, [performance.loc[(performance['dataset'] == dataset) & (performance['detector'] == d) & (performance['label'] == label)]['AP'].mean() for d in detectors],
               width=0.12, color=color, edgecolor='#000000',alpha=0.9,
               yerr=[performance.loc[(performance['dataset'] == dataset) & (performance['detector'] == d) & (performance['label'] == label)]['AP'].std() for d in detectors],
               linewidth=2)



    ax.set_title(f'Results in the real world Over models - {label_name}', fontsize=18)

    ax.set_ylim([0, 110])

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
        #ax.set_xticklabels(model_names, fontsize=17)
        ax.set_xlabel('Qualification rounds', fontsize=17)

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
    text_file = open(f"../latex_plots/qualified_detector_dataset_label_{label_count}.tex", "w")

    #write string to file
    text_file.write(chart_code)

    #close file
    text_file.close()

    #tikzplotlib.save(f"../latex_plots/general_detector_{label}_e{env_number}.tex", axis_height='7cm', axis_width='12cm')
    plt.show()





