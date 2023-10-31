import matplotlib
import numpy as np
import pandas
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.pyplot import subplots
import tikzplotlib

iou_threshold = 0.5
confidence_threshold = 0.75

houses = pd.read_excel('./../../results/faster_rcnn_ap_real_data.xlsx')
houses['AP'] = houses['AP'].astype(np.float64)
houses['AP'] = houses['AP'].apply(lambda x: x*100).round()
houses = houses.loc[(houses['epochs_gd'] == 60) & ((houses['epochs_qd'] == 40) | (houses['epochs_qd'] == 60)) &
                    (houses['iou_threshold'] == iou_threshold) & (houses['confidence_threshold'] == confidence_threshold) &
                    (houses['detector'] == 'GD')]


houses_faster = houses[['dataset', 'label', 'AP',]]
concatenation = houses_faster.copy()
houses_faster = houses_faster.groupby(['dataset', 'label', ], as_index=False)


# YOLO
houses = pd.read_excel('./../../results/yolov5_ap_real_data.xlsx')
houses['AP'] = houses['AP'].astype(np.float64)
houses['AP'] = houses['AP'].apply(lambda x: x*100).round()
houses = houses.loc[(houses['epochs_gd'] == 60) & ((houses['epochs_qd'] == 40) | (houses['epochs_qd'] == 60)) &
                    (houses['iou_threshold'] == iou_threshold) & (houses['confidence_threshold'] == confidence_threshold) & (houses['detector'] == 'GD')]

houses_yolo = houses[['dataset', 'label', 'AP']]
concatenation = pd.concat([concatenation, houses_yolo], ignore_index=True)
houses_yolo = houses_yolo.groupby(['dataset', 'label', ], as_index=False)



# DETR
houses = pd.read_excel('./../../results/detr_ap_real_data.xlsx')
houses['AP'] = houses['AP'].astype(np.float64)
houses['AP'] = houses['AP'].apply(lambda x: x*100).round()
houses = houses.loc[(houses['epochs_gd'] == 60) & ((houses['epochs_qd'] == 40) | (houses['epochs_qd'] == 60)) &
                    (houses['iou_threshold'] == iou_threshold) & (houses['confidence_threshold'] == confidence_threshold) & (houses['detector'] == 'GD')]


houses['dataset'] = houses['dataset'].str.lower()
houses_detr = houses[['dataset', 'label', 'AP']]
concatenation = pd.concat([concatenation, houses_detr], ignore_index=True)
houses_detr = houses_detr.groupby(['dataset', 'label', ], as_index=False)


concatenation = concatenation.groupby(['label', 'dataset'], as_index=False)

model_names = ['DETR~\cite{detr}', 'YOLOv5~\cite{yolo}', 'Faster~R--CNN~\cite{fasterrcnn}', 'Average']
datasets = ['deep_doors_2', 'gibson', 'gibson_deep_doors_2']
datasets_name = ['DD2~\cite{deepdoors2}', 'Gibson', 'Gibson + DD2~\cite{deepdoors2}']

#print(houses_detr.mean().index.tolist())

table = ''

for i, (dataset, dataset_name) in enumerate(zip(datasets, datasets_name)):
    table += '\multicolumn{1}{c|}{\multirow{2}{*}{' + dataset_name + '}} & '
    for c, (label_name, label) in enumerate(zip(['Closed', 'Open',], [0, 1])):
        if c != 0:
            table += '\multicolumn{1}{c|}{} & '
        table += label_name
        for model_name, mean, std in zip(model_names,
                                        [houses_detr.mean(), houses_yolo.mean(), houses_faster.mean(), concatenation.mean()],
                                        [houses_detr.std(), houses_yolo.std(), houses_faster.std(), concatenation.std()]):
            ap = mean.loc[(mean["dataset"] == dataset) & (mean["label"] == label)]['AP'].iloc[0]

            s = std.loc[(std["dataset"] == dataset) & (std["label"] == label)]['AP'].iloc[0]
            table += f'& {round(ap)} & {round(s)}'
        table += '\\\\ [2pt] \n'
    table += '\\hline \n'

# Plots
for label in [0, 1]:
    fig, ax = subplots(figsize=(10, 5))
    dataframes = [houses_detr, houses_yolo, houses_faster, concatenation]

    X = np.arange(4)
    ax.bar(X, [0 for _ in range(4)], width=0.2, label="IGibson")


    for i, dataset in enumerate(datasets):

        ax.bar(X + (i + 1)* 0.2, [d.mean().loc[(d.mean()["dataset"] == dataset) & (d.mean()["label"] == label)]['AP'].iloc[0] for d in dataframes],
               yerr=[d.std().loc[(d.std()["dataset"] == dataset) & (d.std()["label"] == label)]['AP'].iloc[0] for d in dataframes],
               width=0.2, label=datasets_name[i])

    ax.set_title(f'Dataset comparison - Closed doors', fontsize=18)
    ax.set_ylim([0, 40])

    ax.tick_params(axis='y', labelsize=16)
    ax.set_xticks([i+0.3 for i in range(4)])
    ax.set_xticklabels(model_names, fontsize=17)
    ax.set_ylabel('AP', fontsize=17)
    ax.set_xlabel('Environment', fontsize=17)

    ax.legend(prop={"size": 16}, loc='upper center', ncol=4)
    fig.tight_layout()

    def tikzplotlib_fix_ncols(obj):
        """
        workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
        """
        if hasattr(obj, "_ncols"):
            obj._ncol = obj._ncols
        for child in obj.get_children():
            tikzplotlib_fix_ncols(child)
    tikzplotlib_fix_ncols(fig)
    tikzplotlib.save(f"latex_plots/general_detector_{label}.tex", axis_height='7cm', axis_width='12cm')
    plt.show()


print(table)
table=''
for model_name, mean, std in zip(model_names,
                        [houses_detr.mean(), houses_yolo.mean(), houses_faster.mean(), concatenation.mean()],
                        [houses_detr.std(), houses_yolo.std(), houses_faster.std(), concatenation.std()]):
    table += '\multicolumn{1}{c|}{\multirow{2}{*}{' + model_name + '}} & '
    for c, (label_name, label) in enumerate(zip(['Closed', 'Open',], [0, 1])):
        if c != 0:
            table += '\multicolumn{1}{c|}{} & '

        table += label_name + ' & - & - '
        for i, dataset in enumerate(datasets):
            ap = mean.loc[(mean["dataset"] == dataset) & (mean["label"] == label)]['AP'].iloc[0]

            s = std.loc[(std["dataset"] == dataset) & (std["label"] == label)]['AP'].iloc[0]
            table += f'& {round(ap)} & {round(s)}'
        table += '\\\\ [2pt] \n'
    table += '\\hline \n'


print(table)