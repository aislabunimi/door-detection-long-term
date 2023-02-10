import pandas as pd
from matplotlib import pyplot as plt

houses = ['house_1', 'house_2', 'house_7', 'house_9', 'house_10', 'house_13', 'house_15', 'house_20', 'house_21', 'house_22']
epochs_general_detector = [10, 20, 40, 60]
fine_tune_quantity = [25, 50, 75]
legend_labels = ['10 epochs', '20 epochs', '40 epochs', '60 epochs']
titles = ['No door', 'Closed door', 'Open door']

for title, label in zip(titles, [-1, 0, 1]):
    fig, ax = plt.subplots(figsize=(8, 5))
    for label_x, epoch in zip(legend_labels, epochs_general_detector):
        dataframes = [pd.read_excel(f'../../results/gd_epochs_analysis_{epoch}.xlsx')] +\
                     [pd.read_excel(f'../../results/qd_{train_size}_epochs_analysis_{epoch}.xlsx') for train_size in fine_tune_quantity]


        means = []
        for dataframe in dataframes:
            dataframe = dataframe[dataframe['label'] == label]
            means.append(dataframe['AP'].mean())

        ax.plot([0, 1, 2, 3], means, label=label_x)

    ax.set_ylabel('AP')
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(['$GD_{-e}$', '$QD_{e}^{25}$', '$QD_{e}^{50}$', '$QD_{e}^{75}$'])
    ax.set_title(f'AP results for epochs analysis experiment - {title}')
    ax.legend()

    plt.show()
    plt.close()
