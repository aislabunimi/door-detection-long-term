import os

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from doors_detection_long_term.doors_detector.dataset.torch_dataset import DATASET
from doors_detection_long_term.doors_detector.models.generic_model import GenericModel, DESCRIPTION
from doors_detection_long_term.doors_detector.models.model_names import ModelName
from doors_detection_long_term.scripts.doors_detector.dataset_configurator import trained_models_path

TEST: DESCRIPTION = 0

EXP_1_HOUSE_1_40_EPOCHS: DESCRIPTION = 1
EXP_2_HOUSE_1_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_15: DESCRIPTION = 2
EXP_2_HOUSE_1_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_15: DESCRIPTION = 3
EXP_2_HOUSE_1_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_25: DESCRIPTION = 4
EXP_2_HOUSE_1_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_25: DESCRIPTION = 5
EXP_2_HOUSE_1_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_50: DESCRIPTION = 6
EXP_2_HOUSE_1_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_50: DESCRIPTION = 7
EXP_2_HOUSE_1_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_75: DESCRIPTION = 8
EXP_2_HOUSE_1_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_75: DESCRIPTION = 9
EXP_1_HOUSE_1_60_EPOCHS: DESCRIPTION = 10
EXP_2_HOUSE_1_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_15: DESCRIPTION = 11
EXP_2_HOUSE_1_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_15: DESCRIPTION = 12
EXP_2_HOUSE_1_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_25: DESCRIPTION = 13
EXP_2_HOUSE_1_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_25: DESCRIPTION = 14
EXP_2_HOUSE_1_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_50: DESCRIPTION = 15
EXP_2_HOUSE_1_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_50: DESCRIPTION = 16
EXP_2_HOUSE_1_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_75: DESCRIPTION = 17
EXP_2_HOUSE_1_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_75: DESCRIPTION = 18
EXP_1_HOUSE_2_40_EPOCHS: DESCRIPTION = 19
EXP_2_HOUSE_2_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_15: DESCRIPTION = 20
EXP_2_HOUSE_2_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_15: DESCRIPTION = 21
EXP_2_HOUSE_2_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_25: DESCRIPTION = 22
EXP_2_HOUSE_2_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_25: DESCRIPTION = 23
EXP_2_HOUSE_2_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_50: DESCRIPTION = 24
EXP_2_HOUSE_2_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_50: DESCRIPTION = 25
EXP_2_HOUSE_2_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_75: DESCRIPTION = 26
EXP_2_HOUSE_2_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_75: DESCRIPTION = 27
EXP_1_HOUSE_2_60_EPOCHS: DESCRIPTION = 28
EXP_2_HOUSE_2_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_15: DESCRIPTION = 29
EXP_2_HOUSE_2_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_15: DESCRIPTION = 30
EXP_2_HOUSE_2_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_25: DESCRIPTION = 31
EXP_2_HOUSE_2_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_25: DESCRIPTION = 32
EXP_2_HOUSE_2_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_50: DESCRIPTION = 33
EXP_2_HOUSE_2_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_50: DESCRIPTION = 34
EXP_2_HOUSE_2_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_75: DESCRIPTION = 35
EXP_2_HOUSE_2_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_75: DESCRIPTION = 36
EXP_1_HOUSE_7_40_EPOCHS: DESCRIPTION = 37
EXP_2_HOUSE_7_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_15: DESCRIPTION = 38
EXP_2_HOUSE_7_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_15: DESCRIPTION = 39
EXP_2_HOUSE_7_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_25: DESCRIPTION = 40
EXP_2_HOUSE_7_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_25: DESCRIPTION = 41
EXP_2_HOUSE_7_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_50: DESCRIPTION = 42
EXP_2_HOUSE_7_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_50: DESCRIPTION = 43
EXP_2_HOUSE_7_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_75: DESCRIPTION = 44
EXP_2_HOUSE_7_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_75: DESCRIPTION = 45
EXP_1_HOUSE_7_60_EPOCHS: DESCRIPTION = 46
EXP_2_HOUSE_7_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_15: DESCRIPTION = 47
EXP_2_HOUSE_7_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_15: DESCRIPTION = 48
EXP_2_HOUSE_7_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_25: DESCRIPTION = 49
EXP_2_HOUSE_7_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_25: DESCRIPTION = 50
EXP_2_HOUSE_7_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_50: DESCRIPTION = 51
EXP_2_HOUSE_7_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_50: DESCRIPTION = 52
EXP_2_HOUSE_7_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_75: DESCRIPTION = 53
EXP_2_HOUSE_7_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_75: DESCRIPTION = 54
EXP_1_HOUSE_9_40_EPOCHS: DESCRIPTION = 55
EXP_2_HOUSE_9_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_15: DESCRIPTION = 56
EXP_2_HOUSE_9_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_15: DESCRIPTION = 57
EXP_2_HOUSE_9_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_25: DESCRIPTION = 58
EXP_2_HOUSE_9_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_25: DESCRIPTION = 59
EXP_2_HOUSE_9_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_50: DESCRIPTION = 60
EXP_2_HOUSE_9_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_50: DESCRIPTION = 61
EXP_2_HOUSE_9_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_75: DESCRIPTION = 62
EXP_2_HOUSE_9_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_75: DESCRIPTION = 63
EXP_1_HOUSE_9_60_EPOCHS: DESCRIPTION = 64
EXP_2_HOUSE_9_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_15: DESCRIPTION = 65
EXP_2_HOUSE_9_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_15: DESCRIPTION = 66
EXP_2_HOUSE_9_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_25: DESCRIPTION = 67
EXP_2_HOUSE_9_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_25: DESCRIPTION = 68
EXP_2_HOUSE_9_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_50: DESCRIPTION = 69
EXP_2_HOUSE_9_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_50: DESCRIPTION = 70
EXP_2_HOUSE_9_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_75: DESCRIPTION = 71
EXP_2_HOUSE_9_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_75: DESCRIPTION = 72
EXP_1_HOUSE_10_40_EPOCHS: DESCRIPTION = 73
EXP_2_HOUSE_10_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_15: DESCRIPTION = 74
EXP_2_HOUSE_10_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_15: DESCRIPTION = 75
EXP_2_HOUSE_10_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_25: DESCRIPTION = 76
EXP_2_HOUSE_10_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_25: DESCRIPTION = 77
EXP_2_HOUSE_10_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_50: DESCRIPTION = 78
EXP_2_HOUSE_10_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_50: DESCRIPTION = 79
EXP_2_HOUSE_10_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_75: DESCRIPTION = 80
EXP_2_HOUSE_10_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_75: DESCRIPTION = 81
EXP_1_HOUSE_10_60_EPOCHS: DESCRIPTION = 82
EXP_2_HOUSE_10_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_15: DESCRIPTION = 83
EXP_2_HOUSE_10_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_15: DESCRIPTION = 84
EXP_2_HOUSE_10_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_25: DESCRIPTION = 85
EXP_2_HOUSE_10_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_25: DESCRIPTION = 86
EXP_2_HOUSE_10_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_50: DESCRIPTION = 87
EXP_2_HOUSE_10_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_50: DESCRIPTION = 88
EXP_2_HOUSE_10_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_75: DESCRIPTION = 89
EXP_2_HOUSE_10_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_75: DESCRIPTION = 90
EXP_1_HOUSE_13_40_EPOCHS: DESCRIPTION = 91
EXP_2_HOUSE_13_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_15: DESCRIPTION = 92
EXP_2_HOUSE_13_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_15: DESCRIPTION = 93
EXP_2_HOUSE_13_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_25: DESCRIPTION = 94
EXP_2_HOUSE_13_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_25: DESCRIPTION = 95
EXP_2_HOUSE_13_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_50: DESCRIPTION = 96
EXP_2_HOUSE_13_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_50: DESCRIPTION = 97
EXP_2_HOUSE_13_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_75: DESCRIPTION = 98
EXP_2_HOUSE_13_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_75: DESCRIPTION = 99
EXP_1_HOUSE_13_60_EPOCHS: DESCRIPTION = 100
EXP_2_HOUSE_13_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_15: DESCRIPTION = 101
EXP_2_HOUSE_13_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_15: DESCRIPTION = 102
EXP_2_HOUSE_13_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_25: DESCRIPTION = 103
EXP_2_HOUSE_13_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_25: DESCRIPTION = 104
EXP_2_HOUSE_13_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_50: DESCRIPTION = 105
EXP_2_HOUSE_13_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_50: DESCRIPTION = 106
EXP_2_HOUSE_13_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_75: DESCRIPTION = 107
EXP_2_HOUSE_13_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_75: DESCRIPTION = 108
EXP_1_HOUSE_15_40_EPOCHS: DESCRIPTION = 109
EXP_2_HOUSE_15_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_15: DESCRIPTION = 110
EXP_2_HOUSE_15_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_15: DESCRIPTION = 111
EXP_2_HOUSE_15_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_25: DESCRIPTION = 112
EXP_2_HOUSE_15_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_25: DESCRIPTION = 113
EXP_2_HOUSE_15_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_50: DESCRIPTION = 114
EXP_2_HOUSE_15_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_50: DESCRIPTION = 115
EXP_2_HOUSE_15_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_75: DESCRIPTION = 116
EXP_2_HOUSE_15_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_75: DESCRIPTION = 117
EXP_1_HOUSE_15_60_EPOCHS: DESCRIPTION = 118
EXP_2_HOUSE_15_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_15: DESCRIPTION = 119
EXP_2_HOUSE_15_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_15: DESCRIPTION = 120
EXP_2_HOUSE_15_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_25: DESCRIPTION = 121
EXP_2_HOUSE_15_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_25: DESCRIPTION = 122
EXP_2_HOUSE_15_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_50: DESCRIPTION = 123
EXP_2_HOUSE_15_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_50: DESCRIPTION = 124
EXP_2_HOUSE_15_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_75: DESCRIPTION = 125
EXP_2_HOUSE_15_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_75: DESCRIPTION = 126
EXP_1_HOUSE_20_40_EPOCHS: DESCRIPTION = 127
EXP_2_HOUSE_20_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_15: DESCRIPTION = 128
EXP_2_HOUSE_20_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_15: DESCRIPTION = 129
EXP_2_HOUSE_20_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_25: DESCRIPTION = 130
EXP_2_HOUSE_20_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_25: DESCRIPTION = 131
EXP_2_HOUSE_20_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_50: DESCRIPTION = 132
EXP_2_HOUSE_20_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_50: DESCRIPTION = 133
EXP_2_HOUSE_20_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_75: DESCRIPTION = 134
EXP_2_HOUSE_20_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_75: DESCRIPTION = 135
EXP_1_HOUSE_20_60_EPOCHS: DESCRIPTION = 136
EXP_2_HOUSE_20_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_15: DESCRIPTION = 137
EXP_2_HOUSE_20_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_15: DESCRIPTION = 138
EXP_2_HOUSE_20_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_25: DESCRIPTION = 139
EXP_2_HOUSE_20_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_25: DESCRIPTION = 140
EXP_2_HOUSE_20_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_50: DESCRIPTION = 141
EXP_2_HOUSE_20_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_50: DESCRIPTION = 142
EXP_2_HOUSE_20_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_75: DESCRIPTION = 143
EXP_2_HOUSE_20_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_75: DESCRIPTION = 144
EXP_1_HOUSE_21_40_EPOCHS: DESCRIPTION = 145
EXP_2_HOUSE_21_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_15: DESCRIPTION = 146
EXP_2_HOUSE_21_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_15: DESCRIPTION = 147
EXP_2_HOUSE_21_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_25: DESCRIPTION = 148
EXP_2_HOUSE_21_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_25: DESCRIPTION = 149
EXP_2_HOUSE_21_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_50: DESCRIPTION = 150
EXP_2_HOUSE_21_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_50: DESCRIPTION = 151
EXP_2_HOUSE_21_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_75: DESCRIPTION = 152
EXP_2_HOUSE_21_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_75: DESCRIPTION = 153
EXP_1_HOUSE_21_60_EPOCHS: DESCRIPTION = 154
EXP_2_HOUSE_21_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_15: DESCRIPTION = 155
EXP_2_HOUSE_21_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_15: DESCRIPTION = 156
EXP_2_HOUSE_21_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_25: DESCRIPTION = 157
EXP_2_HOUSE_21_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_25: DESCRIPTION = 158
EXP_2_HOUSE_21_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_50: DESCRIPTION = 159
EXP_2_HOUSE_21_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_50: DESCRIPTION = 160
EXP_2_HOUSE_21_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_75: DESCRIPTION = 161
EXP_2_HOUSE_21_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_75: DESCRIPTION = 162
EXP_1_HOUSE_22_40_EPOCHS: DESCRIPTION = 163
EXP_2_HOUSE_22_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_15: DESCRIPTION = 164
EXP_2_HOUSE_22_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_15: DESCRIPTION = 165
EXP_2_HOUSE_22_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_25: DESCRIPTION = 166
EXP_2_HOUSE_22_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_25: DESCRIPTION = 167
EXP_2_HOUSE_22_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_50: DESCRIPTION = 168
EXP_2_HOUSE_22_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_50: DESCRIPTION = 169
EXP_2_HOUSE_22_EPOCHS_GD_40_EPOCH_QD_20_FINE_TUNE_75: DESCRIPTION = 170
EXP_2_HOUSE_22_EPOCHS_GD_40_EPOCH_QD_40_FINE_TUNE_75: DESCRIPTION = 171
EXP_1_HOUSE_22_60_EPOCHS: DESCRIPTION = 172
EXP_2_HOUSE_22_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_15: DESCRIPTION = 173
EXP_2_HOUSE_22_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_15: DESCRIPTION = 174
EXP_2_HOUSE_22_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_25: DESCRIPTION = 175
EXP_2_HOUSE_22_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_25: DESCRIPTION = 176
EXP_2_HOUSE_22_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_50: DESCRIPTION = 177
EXP_2_HOUSE_22_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_50: DESCRIPTION = 178
EXP_2_HOUSE_22_EPOCHS_GD_60_EPOCH_QD_20_FINE_TUNE_75: DESCRIPTION = 179
EXP_2_HOUSE_22_EPOCHS_GD_60_EPOCH_QD_40_FINE_TUNE_75: DESCRIPTION = 180

EXP_GENERAL_DETECTOR_GIBSON_DEEP_DOORS_2_40_EPOCHS: DESCRIPTION = 181
EXP_GENERAL_DETECTOR_GIBSON_DEEP_DOORS_2_60_EPOCHS: DESCRIPTION = 182
EXP_GENERAL_DETECTOR_DEEP_DOORS_2_40_EPOCHS: DESCRIPTION = 183
EXP_GENERAL_DETECTOR_DEEP_DOORS_2_60_EPOCHS: DESCRIPTION = 184
EXP_GENERAL_DETECTOR_GIBSON_40_EPOCHS: DESCRIPTION = 185
EXP_GENERAL_DETECTOR_GIBSON_60_EPOCHS: DESCRIPTION = 186
EXP_2_FLOOR1_GIBSON_DEEP_DOORS_2_EPOCHS_GD_40_EPOCHS_QD_20_FINE_TUNE_15: DESCRIPTION = 187
EXP_2_FLOOR1_GIBSON_DEEP_DOORS_2_EPOCHS_GD_40_EPOCHS_QD_40_FINE_TUNE_15: DESCRIPTION = 188
EXP_2_FLOOR1_GIBSON_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_20_FINE_TUNE_15: DESCRIPTION = 189
EXP_2_FLOOR1_GIBSON_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_15: DESCRIPTION = 190
EXP_2_FLOOR1_GIBSON_DEEP_DOORS_2_EPOCHS_GD_40_EPOCHS_QD_20_FINE_TUNE_25: DESCRIPTION = 191
EXP_2_FLOOR1_GIBSON_DEEP_DOORS_2_EPOCHS_GD_40_EPOCHS_QD_40_FINE_TUNE_25: DESCRIPTION = 192
EXP_2_FLOOR1_GIBSON_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_20_FINE_TUNE_25: DESCRIPTION = 193
EXP_2_FLOOR1_GIBSON_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_25: DESCRIPTION = 194
EXP_2_FLOOR1_GIBSON_DEEP_DOORS_2_EPOCHS_GD_40_EPOCHS_QD_20_FINE_TUNE_50: DESCRIPTION = 195
EXP_2_FLOOR1_GIBSON_DEEP_DOORS_2_EPOCHS_GD_40_EPOCHS_QD_40_FINE_TUNE_50: DESCRIPTION = 196
EXP_2_FLOOR1_GIBSON_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_20_FINE_TUNE_50: DESCRIPTION = 197
EXP_2_FLOOR1_GIBSON_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_50: DESCRIPTION = 198
EXP_2_FLOOR1_GIBSON_DEEP_DOORS_2_EPOCHS_GD_40_EPOCHS_QD_20_FINE_TUNE_75: DESCRIPTION = 199
EXP_2_FLOOR1_GIBSON_DEEP_DOORS_2_EPOCHS_GD_40_EPOCHS_QD_40_FINE_TUNE_75: DESCRIPTION = 200
EXP_2_FLOOR1_GIBSON_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_20_FINE_TUNE_75: DESCRIPTION = 201
EXP_2_FLOOR1_GIBSON_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_75: DESCRIPTION = 202
EXP_2_FLOOR1_DEEP_DOORS_2_EPOCHS_GD_40_EPOCHS_QD_20_FINE_TUNE_15: DESCRIPTION = 203
EXP_2_FLOOR1_DEEP_DOORS_2_EPOCHS_GD_40_EPOCHS_QD_40_FINE_TUNE_15: DESCRIPTION = 204
EXP_2_FLOOR1_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_20_FINE_TUNE_15: DESCRIPTION = 205
EXP_2_FLOOR1_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_15: DESCRIPTION = 206
EXP_2_FLOOR1_DEEP_DOORS_2_EPOCHS_GD_40_EPOCHS_QD_20_FINE_TUNE_25: DESCRIPTION = 207
EXP_2_FLOOR1_DEEP_DOORS_2_EPOCHS_GD_40_EPOCHS_QD_40_FINE_TUNE_25: DESCRIPTION = 208
EXP_2_FLOOR1_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_20_FINE_TUNE_25: DESCRIPTION = 209
EXP_2_FLOOR1_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_25: DESCRIPTION = 210
EXP_2_FLOOR1_DEEP_DOORS_2_EPOCHS_GD_40_EPOCHS_QD_20_FINE_TUNE_50: DESCRIPTION = 211
EXP_2_FLOOR1_DEEP_DOORS_2_EPOCHS_GD_40_EPOCHS_QD_40_FINE_TUNE_50: DESCRIPTION = 212
EXP_2_FLOOR1_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_20_FINE_TUNE_50: DESCRIPTION = 213
EXP_2_FLOOR1_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_50: DESCRIPTION = 214
EXP_2_FLOOR1_DEEP_DOORS_2_EPOCHS_GD_40_EPOCHS_QD_20_FINE_TUNE_75: DESCRIPTION = 215
EXP_2_FLOOR1_DEEP_DOORS_2_EPOCHS_GD_40_EPOCHS_QD_40_FINE_TUNE_75: DESCRIPTION = 216
EXP_2_FLOOR1_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_20_FINE_TUNE_75: DESCRIPTION = 217
EXP_2_FLOOR1_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_75: DESCRIPTION = 218
EXP_2_FLOOR1_GIBSON_EPOCHS_GD_40_EPOCHS_QD_20_FINE_TUNE_15: DESCRIPTION = 219
EXP_2_FLOOR1_GIBSON_EPOCHS_GD_40_EPOCHS_QD_40_FINE_TUNE_15: DESCRIPTION = 220
EXP_2_FLOOR1_GIBSON_EPOCHS_GD_60_EPOCHS_QD_20_FINE_TUNE_15: DESCRIPTION = 221
EXP_2_FLOOR1_GIBSON_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_15: DESCRIPTION = 222
EXP_2_FLOOR1_GIBSON_EPOCHS_GD_40_EPOCHS_QD_20_FINE_TUNE_25: DESCRIPTION = 223
EXP_2_FLOOR1_GIBSON_EPOCHS_GD_40_EPOCHS_QD_40_FINE_TUNE_25: DESCRIPTION = 224
EXP_2_FLOOR1_GIBSON_EPOCHS_GD_60_EPOCHS_QD_20_FINE_TUNE_25: DESCRIPTION = 225
EXP_2_FLOOR1_GIBSON_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_25: DESCRIPTION = 226
EXP_2_FLOOR1_GIBSON_EPOCHS_GD_40_EPOCHS_QD_20_FINE_TUNE_50: DESCRIPTION = 227
EXP_2_FLOOR1_GIBSON_EPOCHS_GD_40_EPOCHS_QD_40_FINE_TUNE_50: DESCRIPTION = 228
EXP_2_FLOOR1_GIBSON_EPOCHS_GD_60_EPOCHS_QD_20_FINE_TUNE_50: DESCRIPTION = 229
EXP_2_FLOOR1_GIBSON_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_50: DESCRIPTION = 230
EXP_2_FLOOR1_GIBSON_EPOCHS_GD_40_EPOCHS_QD_20_FINE_TUNE_75: DESCRIPTION = 231
EXP_2_FLOOR1_GIBSON_EPOCHS_GD_40_EPOCHS_QD_40_FINE_TUNE_75: DESCRIPTION = 232
EXP_2_FLOOR1_GIBSON_EPOCHS_GD_60_EPOCHS_QD_20_FINE_TUNE_75: DESCRIPTION = 233
EXP_2_FLOOR1_GIBSON_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_75: DESCRIPTION = 234
EXP_2_FLOOR4_GIBSON_DEEP_DOORS_2_EPOCHS_GD_40_EPOCHS_QD_20_FINE_TUNE_15: DESCRIPTION = 235
EXP_2_FLOOR4_GIBSON_DEEP_DOORS_2_EPOCHS_GD_40_EPOCHS_QD_40_FINE_TUNE_15: DESCRIPTION = 236
EXP_2_FLOOR4_GIBSON_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_20_FINE_TUNE_15: DESCRIPTION = 237
EXP_2_FLOOR4_GIBSON_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_15: DESCRIPTION = 238
EXP_2_FLOOR4_GIBSON_DEEP_DOORS_2_EPOCHS_GD_40_EPOCHS_QD_20_FINE_TUNE_25: DESCRIPTION = 239
EXP_2_FLOOR4_GIBSON_DEEP_DOORS_2_EPOCHS_GD_40_EPOCHS_QD_40_FINE_TUNE_25: DESCRIPTION = 240
EXP_2_FLOOR4_GIBSON_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_20_FINE_TUNE_25: DESCRIPTION = 241
EXP_2_FLOOR4_GIBSON_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_25: DESCRIPTION = 242
EXP_2_FLOOR4_GIBSON_DEEP_DOORS_2_EPOCHS_GD_40_EPOCHS_QD_20_FINE_TUNE_50: DESCRIPTION = 243
EXP_2_FLOOR4_GIBSON_DEEP_DOORS_2_EPOCHS_GD_40_EPOCHS_QD_40_FINE_TUNE_50: DESCRIPTION = 244
EXP_2_FLOOR4_GIBSON_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_20_FINE_TUNE_50: DESCRIPTION = 245
EXP_2_FLOOR4_GIBSON_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_50: DESCRIPTION = 246
EXP_2_FLOOR4_GIBSON_DEEP_DOORS_2_EPOCHS_GD_40_EPOCHS_QD_20_FINE_TUNE_75: DESCRIPTION = 247
EXP_2_FLOOR4_GIBSON_DEEP_DOORS_2_EPOCHS_GD_40_EPOCHS_QD_40_FINE_TUNE_75: DESCRIPTION = 248
EXP_2_FLOOR4_GIBSON_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_20_FINE_TUNE_75: DESCRIPTION = 249
EXP_2_FLOOR4_GIBSON_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_75: DESCRIPTION = 250
EXP_2_FLOOR4_DEEP_DOORS_2_EPOCHS_GD_40_EPOCHS_QD_20_FINE_TUNE_15: DESCRIPTION = 251
EXP_2_FLOOR4_DEEP_DOORS_2_EPOCHS_GD_40_EPOCHS_QD_40_FINE_TUNE_15: DESCRIPTION = 252
EXP_2_FLOOR4_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_20_FINE_TUNE_15: DESCRIPTION = 253
EXP_2_FLOOR4_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_15: DESCRIPTION = 254
EXP_2_FLOOR4_DEEP_DOORS_2_EPOCHS_GD_40_EPOCHS_QD_20_FINE_TUNE_25: DESCRIPTION = 255
EXP_2_FLOOR4_DEEP_DOORS_2_EPOCHS_GD_40_EPOCHS_QD_40_FINE_TUNE_25: DESCRIPTION = 256
EXP_2_FLOOR4_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_20_FINE_TUNE_25: DESCRIPTION = 257
EXP_2_FLOOR4_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_25: DESCRIPTION = 258
EXP_2_FLOOR4_DEEP_DOORS_2_EPOCHS_GD_40_EPOCHS_QD_20_FINE_TUNE_50: DESCRIPTION = 259
EXP_2_FLOOR4_DEEP_DOORS_2_EPOCHS_GD_40_EPOCHS_QD_40_FINE_TUNE_50: DESCRIPTION = 260
EXP_2_FLOOR4_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_20_FINE_TUNE_50: DESCRIPTION = 261
EXP_2_FLOOR4_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_50: DESCRIPTION = 262
EXP_2_FLOOR4_DEEP_DOORS_2_EPOCHS_GD_40_EPOCHS_QD_20_FINE_TUNE_75: DESCRIPTION = 263
EXP_2_FLOOR4_DEEP_DOORS_2_EPOCHS_GD_40_EPOCHS_QD_40_FINE_TUNE_75: DESCRIPTION = 264
EXP_2_FLOOR4_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_20_FINE_TUNE_75: DESCRIPTION = 265
EXP_2_FLOOR4_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_75: DESCRIPTION = 266
EXP_2_FLOOR4_GIBSON_EPOCHS_GD_40_EPOCHS_QD_20_FINE_TUNE_15: DESCRIPTION = 267
EXP_2_FLOOR4_GIBSON_EPOCHS_GD_40_EPOCHS_QD_40_FINE_TUNE_15: DESCRIPTION = 268
EXP_2_FLOOR4_GIBSON_EPOCHS_GD_60_EPOCHS_QD_20_FINE_TUNE_15: DESCRIPTION = 269
EXP_2_FLOOR4_GIBSON_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_15: DESCRIPTION = 270
EXP_2_FLOOR4_GIBSON_EPOCHS_GD_40_EPOCHS_QD_20_FINE_TUNE_25: DESCRIPTION = 271
EXP_2_FLOOR4_GIBSON_EPOCHS_GD_40_EPOCHS_QD_40_FINE_TUNE_25: DESCRIPTION = 272
EXP_2_FLOOR4_GIBSON_EPOCHS_GD_60_EPOCHS_QD_20_FINE_TUNE_25: DESCRIPTION = 273
EXP_2_FLOOR4_GIBSON_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_25: DESCRIPTION = 274
EXP_2_FLOOR4_GIBSON_EPOCHS_GD_40_EPOCHS_QD_20_FINE_TUNE_50: DESCRIPTION = 275
EXP_2_FLOOR4_GIBSON_EPOCHS_GD_40_EPOCHS_QD_40_FINE_TUNE_50: DESCRIPTION = 276
EXP_2_FLOOR4_GIBSON_EPOCHS_GD_60_EPOCHS_QD_20_FINE_TUNE_50: DESCRIPTION = 277
EXP_2_FLOOR4_GIBSON_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_50: DESCRIPTION = 278
EXP_2_FLOOR4_GIBSON_EPOCHS_GD_40_EPOCHS_QD_20_FINE_TUNE_75: DESCRIPTION = 279
EXP_2_FLOOR4_GIBSON_EPOCHS_GD_40_EPOCHS_QD_40_FINE_TUNE_75: DESCRIPTION = 280
EXP_2_FLOOR4_GIBSON_EPOCHS_GD_60_EPOCHS_QD_20_FINE_TUNE_75: DESCRIPTION = 281
EXP_2_FLOOR4_GIBSON_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_75: DESCRIPTION = 282
EXP_2_CHEMISTRY_FLOOR0_GIBSON_DEEP_DOORS_2_EPOCHS_GD_40_EPOCHS_QD_20_FINE_TUNE_15: DESCRIPTION = 283
EXP_2_CHEMISTRY_FLOOR0_GIBSON_DEEP_DOORS_2_EPOCHS_GD_40_EPOCHS_QD_40_FINE_TUNE_15: DESCRIPTION = 284
EXP_2_CHEMISTRY_FLOOR0_GIBSON_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_20_FINE_TUNE_15: DESCRIPTION = 285
EXP_2_CHEMISTRY_FLOOR0_GIBSON_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_15: DESCRIPTION = 286
EXP_2_CHEMISTRY_FLOOR0_GIBSON_DEEP_DOORS_2_EPOCHS_GD_40_EPOCHS_QD_20_FINE_TUNE_25: DESCRIPTION = 287
EXP_2_CHEMISTRY_FLOOR0_GIBSON_DEEP_DOORS_2_EPOCHS_GD_40_EPOCHS_QD_40_FINE_TUNE_25: DESCRIPTION = 288
EXP_2_CHEMISTRY_FLOOR0_GIBSON_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_20_FINE_TUNE_25: DESCRIPTION = 289
EXP_2_CHEMISTRY_FLOOR0_GIBSON_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_25: DESCRIPTION = 290
EXP_2_CHEMISTRY_FLOOR0_GIBSON_DEEP_DOORS_2_EPOCHS_GD_40_EPOCHS_QD_20_FINE_TUNE_50: DESCRIPTION = 291
EXP_2_CHEMISTRY_FLOOR0_GIBSON_DEEP_DOORS_2_EPOCHS_GD_40_EPOCHS_QD_40_FINE_TUNE_50: DESCRIPTION = 292
EXP_2_CHEMISTRY_FLOOR0_GIBSON_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_20_FINE_TUNE_50: DESCRIPTION = 293
EXP_2_CHEMISTRY_FLOOR0_GIBSON_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_50: DESCRIPTION = 294
EXP_2_CHEMISTRY_FLOOR0_GIBSON_DEEP_DOORS_2_EPOCHS_GD_40_EPOCHS_QD_20_FINE_TUNE_75: DESCRIPTION = 295
EXP_2_CHEMISTRY_FLOOR0_GIBSON_DEEP_DOORS_2_EPOCHS_GD_40_EPOCHS_QD_40_FINE_TUNE_75: DESCRIPTION = 296
EXP_2_CHEMISTRY_FLOOR0_GIBSON_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_20_FINE_TUNE_75: DESCRIPTION = 297
EXP_2_CHEMISTRY_FLOOR0_GIBSON_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_75: DESCRIPTION = 298
EXP_2_CHEMISTRY_FLOOR0_DEEP_DOORS_2_EPOCHS_GD_40_EPOCHS_QD_20_FINE_TUNE_15: DESCRIPTION = 299
EXP_2_CHEMISTRY_FLOOR0_DEEP_DOORS_2_EPOCHS_GD_40_EPOCHS_QD_40_FINE_TUNE_15: DESCRIPTION = 300
EXP_2_CHEMISTRY_FLOOR0_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_20_FINE_TUNE_15: DESCRIPTION = 301
EXP_2_CHEMISTRY_FLOOR0_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_15: DESCRIPTION = 302
EXP_2_CHEMISTRY_FLOOR0_DEEP_DOORS_2_EPOCHS_GD_40_EPOCHS_QD_20_FINE_TUNE_25: DESCRIPTION = 303
EXP_2_CHEMISTRY_FLOOR0_DEEP_DOORS_2_EPOCHS_GD_40_EPOCHS_QD_40_FINE_TUNE_25: DESCRIPTION = 304
EXP_2_CHEMISTRY_FLOOR0_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_20_FINE_TUNE_25: DESCRIPTION = 305
EXP_2_CHEMISTRY_FLOOR0_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_25: DESCRIPTION = 306
EXP_2_CHEMISTRY_FLOOR0_DEEP_DOORS_2_EPOCHS_GD_40_EPOCHS_QD_20_FINE_TUNE_50: DESCRIPTION = 307
EXP_2_CHEMISTRY_FLOOR0_DEEP_DOORS_2_EPOCHS_GD_40_EPOCHS_QD_40_FINE_TUNE_50: DESCRIPTION = 308
EXP_2_CHEMISTRY_FLOOR0_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_20_FINE_TUNE_50: DESCRIPTION = 309
EXP_2_CHEMISTRY_FLOOR0_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_50: DESCRIPTION = 310
EXP_2_CHEMISTRY_FLOOR0_DEEP_DOORS_2_EPOCHS_GD_40_EPOCHS_QD_20_FINE_TUNE_75: DESCRIPTION = 311
EXP_2_CHEMISTRY_FLOOR0_DEEP_DOORS_2_EPOCHS_GD_40_EPOCHS_QD_40_FINE_TUNE_75: DESCRIPTION = 312
EXP_2_CHEMISTRY_FLOOR0_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_20_FINE_TUNE_75: DESCRIPTION = 313
EXP_2_CHEMISTRY_FLOOR0_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_75: DESCRIPTION = 314
EXP_2_CHEMISTRY_FLOOR0_GIBSON_EPOCHS_GD_40_EPOCHS_QD_20_FINE_TUNE_15: DESCRIPTION = 315
EXP_2_CHEMISTRY_FLOOR0_GIBSON_EPOCHS_GD_40_EPOCHS_QD_40_FINE_TUNE_15: DESCRIPTION = 316
EXP_2_CHEMISTRY_FLOOR0_GIBSON_EPOCHS_GD_60_EPOCHS_QD_20_FINE_TUNE_15: DESCRIPTION = 317
EXP_2_CHEMISTRY_FLOOR0_GIBSON_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_15: DESCRIPTION = 318
EXP_2_CHEMISTRY_FLOOR0_GIBSON_EPOCHS_GD_40_EPOCHS_QD_20_FINE_TUNE_25: DESCRIPTION = 319
EXP_2_CHEMISTRY_FLOOR0_GIBSON_EPOCHS_GD_40_EPOCHS_QD_40_FINE_TUNE_25: DESCRIPTION = 320
EXP_2_CHEMISTRY_FLOOR0_GIBSON_EPOCHS_GD_60_EPOCHS_QD_20_FINE_TUNE_25: DESCRIPTION = 321
EXP_2_CHEMISTRY_FLOOR0_GIBSON_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_25: DESCRIPTION = 322
EXP_2_CHEMISTRY_FLOOR0_GIBSON_EPOCHS_GD_40_EPOCHS_QD_20_FINE_TUNE_50: DESCRIPTION = 323
EXP_2_CHEMISTRY_FLOOR0_GIBSON_EPOCHS_GD_40_EPOCHS_QD_40_FINE_TUNE_50: DESCRIPTION = 324
EXP_2_CHEMISTRY_FLOOR0_GIBSON_EPOCHS_GD_60_EPOCHS_QD_20_FINE_TUNE_50: DESCRIPTION = 325
EXP_2_CHEMISTRY_FLOOR0_GIBSON_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_50: DESCRIPTION = 326
EXP_2_CHEMISTRY_FLOOR0_GIBSON_EPOCHS_GD_40_EPOCHS_QD_20_FINE_TUNE_75: DESCRIPTION = 327
EXP_2_CHEMISTRY_FLOOR0_GIBSON_EPOCHS_GD_40_EPOCHS_QD_40_FINE_TUNE_75: DESCRIPTION = 328
EXP_2_CHEMISTRY_FLOOR0_GIBSON_EPOCHS_GD_60_EPOCHS_QD_20_FINE_TUNE_75: DESCRIPTION = 329
EXP_2_CHEMISTRY_FLOOR0_GIBSON_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_75: DESCRIPTION = 330

##IGibson
# Principal Models
EXP_GENERAL_DETECTOR_IGIBSON_20_EPOCHS: DESCRIPTION = 331
EXP_GENERAL_DETECTOR_IGIBSON_40_EPOCHS: DESCRIPTION = 332
EXP_GENERAL_DETECTOR_IGIBSON_60_EPOCHS: DESCRIPTION = 333

# Finetune house matteo
EXP_2_HOUSE_MATTEO_GIBSON_EPOCHS_GD_40_EPOCHS_QD_20_FINE_TUNE_15: DESCRIPTION = 334
EXP_2_HOUSE_MATTEO_GIBSON_EPOCHS_GD_40_EPOCHS_QD_40_FINE_TUNE_15: DESCRIPTION = 335
EXP_2_HOUSE_MATTEO_GIBSON_EPOCHS_GD_40_EPOCHS_QD_20_FINE_TUNE_25: DESCRIPTION = 336
EXP_2_HOUSE_MATTEO_GIBSON_EPOCHS_GD_40_EPOCHS_QD_40_FINE_TUNE_25: DESCRIPTION = 337
EXP_2_HOUSE_MATTEO_GIBSON_EPOCHS_GD_40_EPOCHS_QD_20_FINE_TUNE_50: DESCRIPTION = 338
EXP_2_HOUSE_MATTEO_GIBSON_EPOCHS_GD_40_EPOCHS_QD_40_FINE_TUNE_50: DESCRIPTION = 339
EXP_2_HOUSE_MATTEO_GIBSON_EPOCHS_GD_40_EPOCHS_QD_20_FINE_TUNE_75: DESCRIPTION = 340
EXP_2_HOUSE_MATTEO_GIBSON_EPOCHS_GD_40_EPOCHS_QD_40_FINE_TUNE_75: DESCRIPTION = 341
EXP_2_HOUSE_MATTEO_GIBSON_EPOCHS_GD_60_EPOCHS_QD_20_FINE_TUNE_15: DESCRIPTION = 342
EXP_2_HOUSE_MATTEO_GIBSON_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_15: DESCRIPTION = 343
EXP_2_HOUSE_MATTEO_GIBSON_EPOCHS_GD_60_EPOCHS_QD_20_FINE_TUNE_25: DESCRIPTION = 344
EXP_2_HOUSE_MATTEO_GIBSON_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_25: DESCRIPTION = 345
EXP_2_HOUSE_MATTEO_GIBSON_EPOCHS_GD_60_EPOCHS_QD_20_FINE_TUNE_50: DESCRIPTION = 346
EXP_2_HOUSE_MATTEO_GIBSON_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_50: DESCRIPTION = 347
EXP_2_HOUSE_MATTEO_GIBSON_EPOCHS_GD_60_EPOCHS_QD_20_FINE_TUNE_75: DESCRIPTION = 348
EXP_2_HOUSE_MATTEO_GIBSON_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_75: DESCRIPTION = 349
EXP_2_HOUSE_MATTEO_DEEP_DOORS_2_EPOCHS_GD_40_EPOCHS_QD_20_FINE_TUNE_15: DESCRIPTION = 350
EXP_2_HOUSE_MATTEO_DEEP_DOORS_2_EPOCHS_GD_40_EPOCHS_QD_40_FINE_TUNE_15: DESCRIPTION = 351
EXP_2_HOUSE_MATTEO_DEEP_DOORS_2_EPOCHS_GD_40_EPOCHS_QD_20_FINE_TUNE_25: DESCRIPTION = 352
EXP_2_HOUSE_MATTEO_DEEP_DOORS_2_EPOCHS_GD_40_EPOCHS_QD_40_FINE_TUNE_25: DESCRIPTION = 353
EXP_2_HOUSE_MATTEO_DEEP_DOORS_2_EPOCHS_GD_40_EPOCHS_QD_20_FINE_TUNE_50: DESCRIPTION = 354
EXP_2_HOUSE_MATTEO_DEEP_DOORS_2_EPOCHS_GD_40_EPOCHS_QD_40_FINE_TUNE_50: DESCRIPTION = 355
EXP_2_HOUSE_MATTEO_DEEP_DOORS_2_EPOCHS_GD_40_EPOCHS_QD_20_FINE_TUNE_75: DESCRIPTION = 356
EXP_2_HOUSE_MATTEO_DEEP_DOORS_2_EPOCHS_GD_40_EPOCHS_QD_40_FINE_TUNE_75: DESCRIPTION = 357
EXP_2_HOUSE_MATTEO_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_20_FINE_TUNE_15: DESCRIPTION = 358
EXP_2_HOUSE_MATTEO_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_15: DESCRIPTION = 359
EXP_2_HOUSE_MATTEO_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_20_FINE_TUNE_25: DESCRIPTION = 360
EXP_2_HOUSE_MATTEO_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_25: DESCRIPTION = 361
EXP_2_HOUSE_MATTEO_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_20_FINE_TUNE_50: DESCRIPTION = 362
EXP_2_HOUSE_MATTEO_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_50: DESCRIPTION = 363
EXP_2_HOUSE_MATTEO_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_20_FINE_TUNE_75: DESCRIPTION = 364
EXP_2_HOUSE_MATTEO_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_75: DESCRIPTION = 365
EXP_2_HOUSE_MATTEO_GIBSON_DEEP_DOORS_2_EPOCHS_GD_40_EPOCHS_QD_20_FINE_TUNE_15: DESCRIPTION = 366
EXP_2_HOUSE_MATTEO_GIBSON_DEEP_DOORS_2_EPOCHS_GD_40_EPOCHS_QD_40_FINE_TUNE_15: DESCRIPTION = 367
EXP_2_HOUSE_MATTEO_GIBSON_DEEP_DOORS_2_EPOCHS_GD_40_EPOCHS_QD_20_FINE_TUNE_25: DESCRIPTION = 368
EXP_2_HOUSE_MATTEO_GIBSON_DEEP_DOORS_2_EPOCHS_GD_40_EPOCHS_QD_40_FINE_TUNE_25: DESCRIPTION = 369
EXP_2_HOUSE_MATTEO_GIBSON_DEEP_DOORS_2_EPOCHS_GD_40_EPOCHS_QD_20_FINE_TUNE_50: DESCRIPTION = 370
EXP_2_HOUSE_MATTEO_GIBSON_DEEP_DOORS_2_EPOCHS_GD_40_EPOCHS_QD_40_FINE_TUNE_50: DESCRIPTION = 371
EXP_2_HOUSE_MATTEO_GIBSON_DEEP_DOORS_2_EPOCHS_GD_40_EPOCHS_QD_20_FINE_TUNE_75: DESCRIPTION = 372
EXP_2_HOUSE_MATTEO_GIBSON_DEEP_DOORS_2_EPOCHS_GD_40_EPOCHS_QD_40_FINE_TUNE_75: DESCRIPTION = 373
EXP_2_HOUSE_MATTEO_GIBSON_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_20_FINE_TUNE_15: DESCRIPTION = 374
EXP_2_HOUSE_MATTEO_GIBSON_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_15: DESCRIPTION = 375
EXP_2_HOUSE_MATTEO_GIBSON_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_20_FINE_TUNE_25: DESCRIPTION = 376
EXP_2_HOUSE_MATTEO_GIBSON_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_25: DESCRIPTION = 377
EXP_2_HOUSE_MATTEO_GIBSON_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_20_FINE_TUNE_50: DESCRIPTION = 378
EXP_2_HOUSE_MATTEO_GIBSON_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_50: DESCRIPTION = 379
EXP_2_HOUSE_MATTEO_GIBSON_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_20_FINE_TUNE_75: DESCRIPTION = 380
EXP_2_HOUSE_MATTEO_GIBSON_DEEP_DOORS_2_EPOCHS_GD_60_EPOCHS_QD_40_FINE_TUNE_75: DESCRIPTION = 381

class FasterRCNN(GenericModel):
    def __init__(self, model_name: ModelName, n_labels: int, pretrained: bool, dataset_name: DATASET, description: DESCRIPTION):
        """

        :param model_name: the name of the detr base model
        :param n_labels: the labels' number
        :param pretrained: it refers to the DetrDoorDetector class, not to detr base model.
                            It True, the DetrDoorDetector's weights are loaded, otherwise the weights are loaded only for the detr base model
        """
        super(FasterRCNN, self).__init__(model_name, dataset_name, description)

        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

        # get number of input features for the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_labels)

        if pretrained:
            path = os.path.join('train_params', self._model_name + '_' + str(self._description), str(self._dataset_name))
            if trained_models_path == "":
                path = os.path.join(os.path.dirname(__file__), path)
            else:
                path = os.path.join(trained_models_path, path)
            self.model.load_state_dict(torch.load(os.path.join(path, 'model.pth'), map_location=torch.device('cpu')))

    def forward(self, x, y):

        x = self.model(x, y)
        return x

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    def to(self, device):
        self.model.to(device)


def apply_nms(orig_prediction, iou_thresh=0.1, confidence_threshold=0.75):
    keep = orig_prediction['scores'] >= confidence_threshold
    # torchvision returns the indices of the bboxes to keep
    orig_prediction['boxes'] = orig_prediction['boxes'][keep]
    orig_prediction['scores'] = orig_prediction['scores'][keep]
    orig_prediction['labels'] = orig_prediction['labels'][keep]

    keep = torchvision.ops.nms(orig_prediction['boxes'], orig_prediction['scores'], iou_thresh)

    final_prediction = orig_prediction
    final_prediction['boxes'] = final_prediction['boxes'][keep]
    final_prediction['scores'] = final_prediction['scores'][keep]
    final_prediction['labels'] = final_prediction['labels'][keep]

    return final_prediction
