from typing import Dict, List

import matplotlib.pyplot as plt


def plot_losses(logs: Dict[str, List[Dict[str, float]]], save_to_file: bool = False):

    train_logs = logs['train']
    test_logs = logs['test']

    losses = list(train_logs[0].keys())

    for loss in losses:
        plt.title(loss.capitalize())

        # Print train logs
        plt.plot([i for i in range(len(train_logs))], [log_epoch[loss] for log_epoch in train_logs], label='Train loss')

        # Print test logs
        plt.plot([i for i in range(len(test_logs))], [log_epoch[loss] for log_epoch in test_logs], label='Test loss')
        plt.legend()
        plt.show()