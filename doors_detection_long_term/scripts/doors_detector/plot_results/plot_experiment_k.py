import pandas as pd
from matplotlib import pyplot as plt

from doors_detection_long_term.doors_detector.experiment_k.criterion import CriterionType

houses = ['house1', 'house2', 'house7', 'house9', 'house10', 'house13', 'house15', 'house20', 'house21', 'house22']

criterions = [CriterionType.MIN]
thresholds = ['0.6', '0.75', '0.9']

# Plot closed doors
labels = [0, 1]

for label in labels:
    legend_labels = []
    fig, ax = plt.subplots(figsize=(8, 5))

    for criterion, threshold in [(c, t) for c in criterions for t in thresholds]:

        dataframes = [pd.read_excel(f'../results/{house}_experimentk_{str(criterion)}_{threshold}.xlsx') for house in houses]

        dataframe = dataframes[0]

        for d in dataframes[1:]:
            dataframe = dataframe.append(d, ignore_index=True)

        print(dataframe.columns)
        mean = dataframe.groupby(['Percentage', 'Label']).mean().loc[pd.IndexSlice[:, label], :]
        x = mean.index.get_level_values(0).to_series().round(1).tolist()
        y = mean.AP.tolist()

        l = f'${str(criterion)}_{{{threshold}}}$'
        legend_labels.append(l)
        ax.plot(x, y, label=l)

    ax.set_ylabel('AP')
    ax.set_ylim([0, 1.1])
    if label == 1:
        print(mean)
    if label == 0:
        ax.set_title(f'AP results over all houses - {str(criterion)} - closed doors (0)')
    elif label == 1:
        ax.set_title(f'AP results over all houses - {str(criterion)} - open doors (1)')
    #ax.set_xticks([i / 10 for i in range(10)])
    #ax.set_xticklabels(houses_list)
    ax.legend()

    fig.tight_layout()

    plt.show()

    plt.close()



