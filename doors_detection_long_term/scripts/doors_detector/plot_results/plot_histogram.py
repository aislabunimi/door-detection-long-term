import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

labels = ['No door (-1)', 'Closed door (0)', 'Open door (1)']

AP_MINUS1 = [68, 74, 76, 79]
AP_0 = [66, 74, 84, 83]
AP_1 = [64, 67, 76, 73]

width = 0.15

x = np.arange(len(labels))

fig, ax = plt.subplots()

rects1 = ax.bar(x - width/2 * 3, [AP_MINUS1[0], AP_0[0], AP_1[0]], width-0.05, label='$GD$')
rects2 = ax.bar(x - width/2, [AP_MINUS1[1], AP_0[1], AP_1[1]], width-0.05, label='$FD_{25}$')
rects3 = ax.bar(x + width/2, [AP_MINUS1[2], AP_0[2], AP_1[2]], width-0.05, label='$FD_{50}$')
rects4 = ax.bar(x + width/2 * 3, [AP_MINUS1[3], AP_0[3], AP_1[3]], width-0.05, label='$FD_{75}$')

ax.set_ylabel('AP')
ax.set_ylim([0, 110])
ax.set_title('House13 AP results')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

fig.tight_layout()

plt.show()