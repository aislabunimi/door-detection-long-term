import matplotlib
import numpy as np
import pandas as pd
import tikzplotlib
from matplotlib import pyplot as plt
from matplotlib.pyplot import subplots

quantity = 0.75
houses = ['floor1', 'floor4', 'chemistry_floor0', 'house_matteo']
metric_ap = pd.read_csv('../../../../results/filternet_results_ap.csv')
metric_complete = pd.read_csv('../../../../results/filternet_results_complete.csv')

metric_ap = metric_ap[metric_ap['quantity'] == quantity]
metric_complete = metric_complete[metric_complete['quantity'] == quantity]

metric_ap = metric_ap.groupby(['model', 'house', 'quantity', 'boxes', 'iou_threshold_matching',
                               'confidence_threshold_tasknet', 'iou_threshold_tasknet',
                               'confidence_threshold_filternet', 'iou_threshold_filternet'], as_index=False).sum()


metric_complete = metric_complete.groupby(['model', 'house', 'quantity', 'boxes', 'iou_threshold_matching',
                                           'confidence_threshold_tasknet', 'iou_threshold_tasknet',
                                           'confidence_threshold_filternet', 'iou_threshold_filternet'], as_index=False).sum()

metric_complete['TP_p'] = metric_complete['TP'] / metric_complete['total_positives']
metric_complete['FP_p'] = metric_complete['FP'] / metric_complete['total_positives']
metric_complete['FPiou_p'] = metric_complete['FPiou'] / metric_complete['total_positives']


colors = ['#1F77B4', '#2CA02C', '#FF7F0E', '#D62728', '#8c564b']
fig, ax = subplots(figsize=(10, 5))
for i, (color, boxes) in enumerate(zip(colors, [0, 10, 30, 50, 100])):
    APs = []
    X = np.arange(5) *1.15

    model = 'tasknet' if boxes == 0 else 'filternet'
    boxes = 100 if boxes == 0 else boxes
    for h in houses:
        print(model, boxes)
        APs.append(metric_ap.loc[(metric_ap['model'] == model) & (metric_ap['boxes'] == boxes)
                                   & (metric_ap['house'] == h), 'AP'].tolist()[0]/2*100)
    APs.append(sum(APs) / len(APs))

    ax.bar(X + i * 0.2 + 0.04, APs,
           width=0.16,  color=color, edgecolor='#000000',alpha=0.9,
           linewidth=2)
    i = 0

    rects = ax.patches

    ax.text(
        rects[-1].get_x() + rects[-1].get_width() / 2, APs[-1] + 1, int(round(APs[-1])), fontsize='x-large', ha="center", va="bottom"
    )




#ax.set_title(f'mAP', fontsize=18)

matplotlib.pyplot.tick_params(left=True)
ax.tick_params(axis='y', labelsize=16)
#ax.set_ylabel('%', fontsize=17)
ax.axhline(y=0.0, linewidth=1, color='black')
ax.set_ylim([0, 79])
ax.set_xticks([i*1.15+0.44 for i in range(5)])
ax.set_xticklabels(['$e_1$', '$e_2$', '$e_3$', '$e_4$', '$\overline{e}$'], fontsize=17)
#ax.set_xlabel('Environment', fontsize=17)

ax.legend(prop={"size": 16}, bbox_to_anchor=(0.5, 0.97), loc='upper center', ncol=4, alignment='left')

def tikzplotlib_fix_ncols(obj):
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)
tikzplotlib_fix_ncols(fig)
chart_code = tikzplotlib.get_tikz_code().replace('\\begin{tikzpicture}', '\\begin{tikzpicture}[scale=0.62]')
chart_code = chart_code.replace('\\begin{axis}[', '\\begin{axis}[\nwidth=12cm,\nheight=8cm,')
chart_code = chart_code.replace('legend style={\n', 'legend cell align={left},\nlegend style={\n/tikz/every even column/.append style={column sep=0.3cm},\n')
chart_code = chart_code.replace('ybar legend', 'area legend')
#chart_code = chart_code.replace('\\end{axis}', '\\input{images/tikz/legend_different_boxes_ap}\n\\end{axis}')
chart_code = chart_code.replace('ytick style={color=black}', 'ytick style={color=black},\nylabel style={rotate=-90}')
#chart_code = chart_code.replace('\\end{axis}', '\\input{graphics/legend_extended_metric_general_detector}\n\\end{axis}')
chart_code = chart_code.replace('scale=0.72', "scale=0.8")
chart_code = chart_code.replace('mark size=3', 'mark size=2')
text_file = open(f"../../latex_plots/filternet_different_boxes_ap.tex", "w")
text_file.write(chart_code)

#close file
text_file.close()
fig.show()


