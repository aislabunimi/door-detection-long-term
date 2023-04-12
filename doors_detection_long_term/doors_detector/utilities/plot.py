from typing import Dict, List

import matplotlib.pyplot as plt


def plot_losses(logs: Dict[str, List[Dict[str, float]]], save_to_file: bool = False):

    train_logs = logs.get('train', None)
    train_after_backpropagation = logs.get('train_after_backpropagation', None)
    validation_logs = logs.get('validation', None)
    test_logs = logs.get('test', None)

    losses = list(train_logs[0].keys())

    for loss in losses:
        plt.title(loss.capitalize())

        # Plot train loss after backprop if exist, otherwise print train loss during training
        if train_after_backpropagation is not None:
            plt.plot([i for i in range(len(train_after_backpropagation))], [log_epoch[loss] for log_epoch in train_after_backpropagation], label='Train loss after backprop')
        else:
            plt.plot([i for i in range(len(train_logs))], [log_epoch[loss] for log_epoch in train_logs], label='Train loss')

        try:
            # Print test logs
            plt.plot([i for i in range(len(test_logs))], [log_epoch[loss] for log_epoch in test_logs], label='Test loss')
        except:
            pass

        # Plot validation loss if exists
        if validation_logs is not None:
            plt.plot([i for i in range(len(validation_logs))], [log_epoch[loss] for log_epoch in validation_logs], label='Validation loss')

        plt.legend()
        if not save_to_file:
            plt.show()
        else:
            plt.savefig('a.svg')