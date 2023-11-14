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
for env_number, house in enumerate(['floor1', 'floor4', 'chemistry_floor0', 'house_matteo']):
    fig, ax = subplots(figsize=(10, 5))
    dataframes = [houses_detr.loc[houses_detr['dataset'] == 'gibson_deep_doors_2'],
                  houses_yolo.loc[houses_yolo['dataset'] == 'gibson_deep_doors_2'],
                  houses_faster.loc[houses_faster['dataset'] == 'gibson_deep_doors_2']]

    X = np.arange(3)
    #ax.bar(X, [0 for _ in range(3)], width=0.16)

    for i, (detector, color) in enumerate(zip(detectors, colors)):
        ax.bar(X + i * 0.16 + 0.02, [d.loc[(d['detector'] == detector) & (d['house'] == house)]['TP_p'].iloc[0] for d in dataframes],
               width=0.12,  color=color, edgecolor='#000000',alpha=0.9,
               linewidth=2)
        ax.bar(X + i * 0.16 + 0.02, [d.loc[(d['detector'] == detector) & (d['house'] == house)]['FP_p'].iloc[0] * -1 for d in dataframes],
               width=0.12, #fc=(0, 0, 0, 0.0),
               color=color, edgecolor='#000000', hatch='/',
               linewidth=2)

        #plt.errorbar(x=X + (i + 1) * 0.2 + 0.04, y=[0.0 for _ in dataframes],
         #            yerr=[[0 for _ in dataframes], [d.loc[(d['dataset'] == dataset) & (d['house'] == house)]['FPiou_p'].iloc[0] for d in dataframes]],
          #           color='#000000', elinewidth=4, capsize=4, capthick=5)

        plt.vlines(x=X + i * 0.16 + 0.02, ymin=[0 for _ in dataframes], ymax=[d.loc[(d['detector'] == detector) & (d['house'] == house)]['FPiou_p'].iloc[0] * -1 for d in dataframes], colors='#000000', ls='-', lw=4)
        plt.plot(X + i * 0.16 + 0.02, [d.loc[(d['detector'] == detector) & (d['house'] == house)]['FPiou_p'].iloc[0] * -1 for d in dataframes], color='#000000', marker='o', linestyle='None')



    ax.set_title(f'Extended metric results in $e_{env_number}$', fontsize=18)
    ax.axhline(y=0.0, linewidth=1, color='black')
    ax.set_ylim([-55, 120])

    if env_number % 2 == 0:
        matplotlib.pyplot.tick_params(left=True)
        ax.tick_params(axis='y', labelsize=16)
        ax.set_ylabel('%', fontsize=17)
    else:
        matplotlib.pyplot.tick_params(left=False)

    if env_number<=1:
        matplotlib.pyplot.tick_params(bottom=False)
    if env_number >1:
        matplotlib.pyplot.tick_params(bottom=True)
        ax.set_xticks([i+0.34 for i in range(3)])
        ax.set_xticklabels(model_names, fontsize=17)
        ax.set_xlabel('Detector', fontsize=17)

    ax.legend(prop={"size": 16}, bbox_to_anchor=(0.5, 0.97), loc='upper center', ncol=4, alignment='left')
    ax.set_yticklabels([item.get_text().replace(chr(8722), '') for item in ax.get_yticklabels()])
    fig.tight_layout()

    def tikzplotlib_fix_ncols(obj):
        if hasattr(obj, "_ncols"):
            obj._ncol = obj._ncols
        for child in obj.get_children():
            tikzplotlib_fix_ncols(child)
    tikzplotlib_fix_ncols(fig)
    chart_code = tikzplotlib.get_tikz_code().replace('\\begin{tikzpicture}', '\\begin{tikzpicture}[scale=0.725]')
    chart_code = chart_code.replace('\\begin{axis}[', '\\begin{axis}[\nwidth=12cm,\nheight=8cm,')
    chart_code = chart_code.replace('legend style={\n', 'legend cell align={left},\nlegend style={\n/tikz/every even column/.append style={column sep=0.3cm},\n')
    chart_code = chart_code.replace('ybar legend', 'area legend')
    #chart_code = chart_code.replace('\\end{axis}', '\\input{graphics/legend_extended_metric_general_detector}\n\\end{axis}')
    chart_code = chart_code.replace('mark size=3', 'mark size=2')
    text_file = open(f"../latex_plots/qualified_detectors_stacked_complete_metric_e{env_number}.tex", "w")

    #write string to file
    text_file.write(chart_code)

    #close file
    text_file.close()

    #tikzplotlib.save(f"../latex_plots/general_detector_{label}_e{env_number}.tex", axis_height='7cm', axis_width='12cm')
    plt.show()

# Plots mean AP
colors = ['#1F77B4','#2CA02C', '#FF7F0E', '#D62728', '#8C564B']
detectors = ['GD', 'QD_15', 'QD_25', 'QD_50', 'QD_75']

for env_number, house in enumerate(['floor1', 'floor4', 'chemistry_floor0', 'house_matteo']):
    fig, ax = subplots(figsize=(10, 5))
    dataframes = [houses_detr.loc[houses_detr['dataset'] == 'gibson_deep_doors_2'],
                  houses_yolo.loc[houses_yolo['dataset'] == 'gibson_deep_doors_2'],
                  houses_faster.loc[houses_faster['dataset'] == 'gibson_deep_doors_2']]

    X = np.arange(3)
    #ax.bar(X, [0 for _ in range(3)], width=0.16)

    for data_count, data in enumerate(dataframes):
        ax.plot([i + 5*data_count for i in range(5)], [data.loc[(data['detector'] == detector) & (data['house'] == house)]['TP_p'].iloc[0] for detector in detectors],
               color='#2CA02C', marker='o',markersize=7)
        ax.fill_between([i + 5*data_count for i in range(5)], [data.loc[(data['detector'] == detector) & (data['house'] == house)]['TP_p'].iloc[0] for detector in detectors], 0,
                        color='#2CA02C', alpha=.2)
        ax.plot([i+ 5*data_count for i in range(5)], [data.loc[(data['detector'] == detector) & (data['house'] == house)]['FP_p'].iloc[0] for detector in detectors],
               color='#D62728', marker='^',markersize=7 )
        ax.fill_between([i+ 5*data_count for i in range(5)], [data.loc[(data['detector'] == detector) & (data['house'] == house)]['FP_p'].iloc[0] for detector in detectors], 0,
                        color='#D62728', alpha=.2)
        ax.plot([i+ 5*data_count for i in range(5)], [data.loc[(data['detector'] == detector) & (data['house'] == house)]['FPiou_p'].iloc[0] for detector in detectors],
               color='#FF7F0E', marker='d',markersize=7)
        ax.fill_between([i+ 5*data_count for i in range(5)], [data.loc[(data['detector'] == detector) & (data['house'] == house)]['FPiou_p'].iloc[0] for detector in detectors], 0,
                        color='#FF7F0E', alpha=.2)


    ax.set_title(f'Extended metric results in $e_{env_number}$', fontsize=18)
    ax.axhline(y=0.0, linewidth=1, color='black')
    ax.set_ylim([0, 100])

    if env_number % 2 == 0:
        matplotlib.pyplot.tick_params(left=True)
        ax.tick_params(axis='y', labelsize=16)
        ax.set_ylabel('%', fontsize=17)
    else:
        matplotlib.pyplot.tick_params(left=False)

    if env_number<=1:
        matplotlib.pyplot.tick_params(bottom=False)
    if env_number >1:
        matplotlib.pyplot.tick_params(bottom=True)
        ax.set_xticks([(i * 5) + 2 for i in range(3)])
        ax.set_xticklabels(model_names, fontsize=17)
        ax.set_xlabel('Detector', fontsize=17)

    ax.legend(prop={"size": 16}, bbox_to_anchor=(0.5, 0.97), loc='upper center', ncol=4, alignment='left')
    ax.set_yticklabels([item.get_text().replace(chr(8722), '') for item in ax.get_yticklabels()])
    fig.tight_layout()

    def tikzplotlib_fix_ncols(obj):
        if hasattr(obj, "_ncols"):
            obj._ncol = obj._ncols
        for child in obj.get_children():
            tikzplotlib_fix_ncols(child)
    tikzplotlib_fix_ncols(fig)
    chart_code = tikzplotlib.get_tikz_code().replace('\\begin{tikzpicture}', '\\begin{tikzpicture}[scale=0.725]')
    chart_code = chart_code.replace('\\begin{axis}[', '\\begin{axis}[\nwidth=12cm,\nheight=8cm,')
    chart_code = chart_code.replace('legend style={\n', 'legend cell align={left},\nlegend style={\n/tikz/every even column/.append style={column sep=0.3cm},\n')
    chart_code = chart_code.replace('ybar legend', 'area legend')
    #chart_code = chart_code.replace('\\end{axis}', '\\input{graphics/legend_extended_metric_general_detector}\n\\end{axis}')
    chart_code = chart_code.replace('mark size=3', 'mark size=2')
    text_file = open(f"../latex_plots/qualified_detectors_stacked_complete_metric_e{env_number}_type_2.tex", "w")

    #write string to file
    text_file.write(chart_code)

    #close file
    text_file.close()

    #tikzplotlib.save(f"../latex_plots/general_detector_{label}_e{env_number}.tex", axis_height='7cm', axis_width='12cm')
    plt.show()