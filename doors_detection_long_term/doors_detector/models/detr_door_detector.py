import os
from typing import Tuple

import torch
from torch import nn

from doors_detection_long_term.doors_detector.dataset.torch_dataset import DATASET
from doors_detection_long_term.doors_detector.models.mlp import MLP
from doors_detection_long_term.doors_detector.models.model_names import ModelName
from doors_detection_long_term.scripts.doors_detector.dataset_configurator import trained_models_path

DESCRIPTION = int


DEEP_DOORS_2_LABELLED_EXP: DESCRIPTION = 0
EXP_1_HOUSE_1: DESCRIPTION = 1
EXP_2_HOUSE_1_25: DESCRIPTION = 2
EXP_2_HOUSE_1_50: DESCRIPTION = 3
EXP_2_HOUSE_1_75: DESCRIPTION = 4
EXP_1_HOUSE_2: DESCRIPTION = 5
EXP_2_HOUSE_2_25: DESCRIPTION = 6
EXP_2_HOUSE_2_50: DESCRIPTION = 7
EXP_2_HOUSE_2_75: DESCRIPTION = 8
EXP_1_HOUSE_7: DESCRIPTION = 9
EXP_2_HOUSE_7_25: DESCRIPTION = 10
EXP_2_HOUSE_7_50: DESCRIPTION = 11
EXP_2_HOUSE_7_75: DESCRIPTION = 12
EXP_1_HOUSE_9: DESCRIPTION = 13
EXP_2_HOUSE_9_25: DESCRIPTION = 14
EXP_2_HOUSE_9_50: DESCRIPTION = 15
EXP_2_HOUSE_9_75: DESCRIPTION = 16
EXP_1_HOUSE_10: DESCRIPTION = 17
EXP_2_HOUSE_10_25: DESCRIPTION = 18
EXP_2_HOUSE_10_50: DESCRIPTION = 19
EXP_2_HOUSE_10_75: DESCRIPTION = 20
EXP_1_HOUSE_13: DESCRIPTION = 21
EXP_2_HOUSE_13_25: DESCRIPTION = 22
EXP_2_HOUSE_13_50: DESCRIPTION = 23
EXP_2_HOUSE_13_75: DESCRIPTION = 24
EXP_1_HOUSE_15: DESCRIPTION = 25
EXP_2_HOUSE_15_25: DESCRIPTION = 26
EXP_2_HOUSE_15_50: DESCRIPTION = 27
EXP_2_HOUSE_15_75: DESCRIPTION = 28
EXP_1_HOUSE_20: DESCRIPTION = 29
EXP_2_HOUSE_20_25: DESCRIPTION = 20
EXP_2_HOUSE_20_50: DESCRIPTION = 31
EXP_2_HOUSE_20_75: DESCRIPTION = 32
EXP_1_HOUSE_21: DESCRIPTION = 33
EXP_2_HOUSE_21_25: DESCRIPTION = 34
EXP_2_HOUSE_21_50: DESCRIPTION = 35
EXP_2_HOUSE_21_75: DESCRIPTION = 36
EXP_1_HOUSE_22: DESCRIPTION = 37
EXP_2_HOUSE_22_25: DESCRIPTION = 38
EXP_2_HOUSE_22_50: DESCRIPTION = 39
EXP_2_HOUSE_22_75: DESCRIPTION = 40

EXP_1_HOUSE_1_EPOCHS_ANALYSIS_10: DESCRIPTION = 41
EXP_1_HOUSE_1_EPOCHS_ANALYSIS_20: DESCRIPTION = 42
EXP_1_HOUSE_1_EPOCHS_ANALYSIS_40: DESCRIPTION = 43
EXP_1_HOUSE_1_EPOCHS_ANALYSIS_60: DESCRIPTION = 44

EXP_2_HOUSE_1_25_EPOCHS_ANALYSIS_10: DESCRIPTION = 45
EXP_2_HOUSE_1_50_EPOCHS_ANALYSIS_10: DESCRIPTION = 46
EXP_2_HOUSE_1_75_EPOCHS_ANALYSIS_10: DESCRIPTION = 47
EXP_2_HOUSE_1_25_EPOCHS_ANALYSIS_20: DESCRIPTION = 48
EXP_2_HOUSE_1_50_EPOCHS_ANALYSIS_20: DESCRIPTION = 49
EXP_2_HOUSE_1_75_EPOCHS_ANALYSIS_20: DESCRIPTION = 50
EXP_2_HOUSE_1_25_EPOCHS_ANALYSIS_40: DESCRIPTION = 51
EXP_2_HOUSE_1_50_EPOCHS_ANALYSIS_40: DESCRIPTION = 52
EXP_2_HOUSE_1_75_EPOCHS_ANALYSIS_40: DESCRIPTION = 53
EXP_2_HOUSE_1_25_EPOCHS_ANALYSIS_60: DESCRIPTION = 54
EXP_2_HOUSE_1_50_EPOCHS_ANALYSIS_60: DESCRIPTION = 55
EXP_2_HOUSE_1_75_EPOCHS_ANALYSIS_60: DESCRIPTION = 56

EXP_1_HOUSE_2_EPOCHS_ANALYSIS_10: DESCRIPTION = 57
EXP_1_HOUSE_2_EPOCHS_ANALYSIS_20: DESCRIPTION = 58
EXP_1_HOUSE_2_EPOCHS_ANALYSIS_40: DESCRIPTION = 59
EXP_1_HOUSE_2_EPOCHS_ANALYSIS_60: DESCRIPTION = 60

EXP_2_HOUSE_2_25_EPOCHS_ANALYSIS_10: DESCRIPTION = 61
EXP_2_HOUSE_2_50_EPOCHS_ANALYSIS_10: DESCRIPTION = 62
EXP_2_HOUSE_2_75_EPOCHS_ANALYSIS_10: DESCRIPTION = 63
EXP_2_HOUSE_2_25_EPOCHS_ANALYSIS_20: DESCRIPTION = 64
EXP_2_HOUSE_2_50_EPOCHS_ANALYSIS_20: DESCRIPTION = 65
EXP_2_HOUSE_2_75_EPOCHS_ANALYSIS_20: DESCRIPTION = 66
EXP_2_HOUSE_2_25_EPOCHS_ANALYSIS_40: DESCRIPTION = 67
EXP_2_HOUSE_2_50_EPOCHS_ANALYSIS_40: DESCRIPTION = 68
EXP_2_HOUSE_2_75_EPOCHS_ANALYSIS_40: DESCRIPTION = 69
EXP_2_HOUSE_2_25_EPOCHS_ANALYSIS_60: DESCRIPTION = 70
EXP_2_HOUSE_2_50_EPOCHS_ANALYSIS_60: DESCRIPTION = 71
EXP_2_HOUSE_2_75_EPOCHS_ANALYSIS_60: DESCRIPTION = 72

EXP_1_HOUSE_7_EPOCHS_ANALYSIS_10: DESCRIPTION = 73
EXP_1_HOUSE_7_EPOCHS_ANALYSIS_20: DESCRIPTION = 74
EXP_1_HOUSE_7_EPOCHS_ANALYSIS_40: DESCRIPTION = 75
EXP_1_HOUSE_7_EPOCHS_ANALYSIS_60: DESCRIPTION = 76

EXP_2_HOUSE_7_25_EPOCHS_ANALYSIS_10: DESCRIPTION = 77
EXP_2_HOUSE_7_50_EPOCHS_ANALYSIS_10: DESCRIPTION = 78
EXP_2_HOUSE_7_75_EPOCHS_ANALYSIS_10: DESCRIPTION = 79
EXP_2_HOUSE_7_25_EPOCHS_ANALYSIS_20: DESCRIPTION = 80
EXP_2_HOUSE_7_50_EPOCHS_ANALYSIS_20: DESCRIPTION = 81
EXP_2_HOUSE_7_75_EPOCHS_ANALYSIS_20: DESCRIPTION = 82
EXP_2_HOUSE_7_25_EPOCHS_ANALYSIS_40: DESCRIPTION = 83
EXP_2_HOUSE_7_50_EPOCHS_ANALYSIS_40: DESCRIPTION = 84
EXP_2_HOUSE_7_75_EPOCHS_ANALYSIS_40: DESCRIPTION = 85
EXP_2_HOUSE_7_25_EPOCHS_ANALYSIS_60: DESCRIPTION = 86
EXP_2_HOUSE_7_50_EPOCHS_ANALYSIS_60: DESCRIPTION = 87
EXP_2_HOUSE_7_75_EPOCHS_ANALYSIS_60: DESCRIPTION = 88

EXP_1_HOUSE_9_EPOCHS_ANALYSIS_10: DESCRIPTION = 89
EXP_1_HOUSE_9_EPOCHS_ANALYSIS_20: DESCRIPTION = 90
EXP_1_HOUSE_9_EPOCHS_ANALYSIS_40: DESCRIPTION = 91
EXP_1_HOUSE_9_EPOCHS_ANALYSIS_60: DESCRIPTION = 92

EXP_2_HOUSE_9_25_EPOCHS_ANALYSIS_10: DESCRIPTION = 93
EXP_2_HOUSE_9_50_EPOCHS_ANALYSIS_10: DESCRIPTION = 94
EXP_2_HOUSE_9_75_EPOCHS_ANALYSIS_10: DESCRIPTION = 95
EXP_2_HOUSE_9_25_EPOCHS_ANALYSIS_20: DESCRIPTION = 96
EXP_2_HOUSE_9_50_EPOCHS_ANALYSIS_20: DESCRIPTION = 97
EXP_2_HOUSE_9_75_EPOCHS_ANALYSIS_20: DESCRIPTION = 98
EXP_2_HOUSE_9_25_EPOCHS_ANALYSIS_40: DESCRIPTION = 99
EXP_2_HOUSE_9_50_EPOCHS_ANALYSIS_40: DESCRIPTION = 100
EXP_2_HOUSE_9_75_EPOCHS_ANALYSIS_40: DESCRIPTION = 101
EXP_2_HOUSE_9_25_EPOCHS_ANALYSIS_60: DESCRIPTION = 102
EXP_2_HOUSE_9_50_EPOCHS_ANALYSIS_60: DESCRIPTION = 103
EXP_2_HOUSE_9_75_EPOCHS_ANALYSIS_60: DESCRIPTION = 104

EXP_1_HOUSE_10_EPOCHS_ANALYSIS_10: DESCRIPTION = 105
EXP_1_HOUSE_10_EPOCHS_ANALYSIS_20: DESCRIPTION = 106
EXP_1_HOUSE_10_EPOCHS_ANALYSIS_40: DESCRIPTION = 107
EXP_1_HOUSE_10_EPOCHS_ANALYSIS_60: DESCRIPTION = 108

EXP_2_HOUSE_10_25_EPOCHS_ANALYSIS_10: DESCRIPTION = 109
EXP_2_HOUSE_10_50_EPOCHS_ANALYSIS_10: DESCRIPTION = 110
EXP_2_HOUSE_10_75_EPOCHS_ANALYSIS_10: DESCRIPTION = 111
EXP_2_HOUSE_10_25_EPOCHS_ANALYSIS_20: DESCRIPTION = 112
EXP_2_HOUSE_10_50_EPOCHS_ANALYSIS_20: DESCRIPTION = 113
EXP_2_HOUSE_10_75_EPOCHS_ANALYSIS_20: DESCRIPTION = 114
EXP_2_HOUSE_10_25_EPOCHS_ANALYSIS_40: DESCRIPTION = 115
EXP_2_HOUSE_10_50_EPOCHS_ANALYSIS_40: DESCRIPTION = 116
EXP_2_HOUSE_10_75_EPOCHS_ANALYSIS_40: DESCRIPTION = 117
EXP_2_HOUSE_10_25_EPOCHS_ANALYSIS_60: DESCRIPTION = 118
EXP_2_HOUSE_10_50_EPOCHS_ANALYSIS_60: DESCRIPTION = 119
EXP_2_HOUSE_10_75_EPOCHS_ANALYSIS_60: DESCRIPTION = 120

EXP_1_HOUSE_13_EPOCHS_ANALYSIS_10: DESCRIPTION = 121
EXP_1_HOUSE_13_EPOCHS_ANALYSIS_20: DESCRIPTION = 122
EXP_1_HOUSE_13_EPOCHS_ANALYSIS_40: DESCRIPTION = 123
EXP_1_HOUSE_13_EPOCHS_ANALYSIS_60: DESCRIPTION = 124

EXP_2_HOUSE_13_25_EPOCHS_ANALYSIS_10: DESCRIPTION = 125
EXP_2_HOUSE_13_50_EPOCHS_ANALYSIS_10: DESCRIPTION = 126
EXP_2_HOUSE_13_75_EPOCHS_ANALYSIS_10: DESCRIPTION = 127
EXP_2_HOUSE_13_25_EPOCHS_ANALYSIS_20: DESCRIPTION = 128
EXP_2_HOUSE_13_50_EPOCHS_ANALYSIS_20: DESCRIPTION = 129
EXP_2_HOUSE_13_75_EPOCHS_ANALYSIS_20: DESCRIPTION = 130
EXP_2_HOUSE_13_25_EPOCHS_ANALYSIS_40: DESCRIPTION = 131
EXP_2_HOUSE_13_50_EPOCHS_ANALYSIS_40: DESCRIPTION = 132
EXP_2_HOUSE_13_75_EPOCHS_ANALYSIS_40: DESCRIPTION = 133
EXP_2_HOUSE_13_25_EPOCHS_ANALYSIS_60: DESCRIPTION = 134
EXP_2_HOUSE_13_50_EPOCHS_ANALYSIS_60: DESCRIPTION = 135
EXP_2_HOUSE_13_75_EPOCHS_ANALYSIS_60: DESCRIPTION = 136

EXP_1_HOUSE_15_EPOCHS_ANALYSIS_10: DESCRIPTION = 137
EXP_1_HOUSE_15_EPOCHS_ANALYSIS_20: DESCRIPTION = 138
EXP_1_HOUSE_15_EPOCHS_ANALYSIS_40: DESCRIPTION = 139
EXP_1_HOUSE_15_EPOCHS_ANALYSIS_60: DESCRIPTION = 140

EXP_2_HOUSE_15_25_EPOCHS_ANALYSIS_10: DESCRIPTION = 141
EXP_2_HOUSE_15_50_EPOCHS_ANALYSIS_10: DESCRIPTION = 142
EXP_2_HOUSE_15_75_EPOCHS_ANALYSIS_10: DESCRIPTION = 143
EXP_2_HOUSE_15_25_EPOCHS_ANALYSIS_20: DESCRIPTION = 144
EXP_2_HOUSE_15_50_EPOCHS_ANALYSIS_20: DESCRIPTION = 145
EXP_2_HOUSE_15_75_EPOCHS_ANALYSIS_20: DESCRIPTION = 146
EXP_2_HOUSE_15_25_EPOCHS_ANALYSIS_40: DESCRIPTION = 147
EXP_2_HOUSE_15_50_EPOCHS_ANALYSIS_40: DESCRIPTION = 148
EXP_2_HOUSE_15_75_EPOCHS_ANALYSIS_40: DESCRIPTION = 149
EXP_2_HOUSE_15_25_EPOCHS_ANALYSIS_60: DESCRIPTION = 150
EXP_2_HOUSE_15_50_EPOCHS_ANALYSIS_60: DESCRIPTION = 151
EXP_2_HOUSE_15_75_EPOCHS_ANALYSIS_60: DESCRIPTION = 152

EXP_1_HOUSE_20_EPOCHS_ANALYSIS_10: DESCRIPTION = 153
EXP_1_HOUSE_20_EPOCHS_ANALYSIS_20: DESCRIPTION = 154
EXP_1_HOUSE_20_EPOCHS_ANALYSIS_40: DESCRIPTION = 155
EXP_1_HOUSE_20_EPOCHS_ANALYSIS_60: DESCRIPTION = 156

EXP_2_HOUSE_20_25_EPOCHS_ANALYSIS_10: DESCRIPTION = 157
EXP_2_HOUSE_20_50_EPOCHS_ANALYSIS_10: DESCRIPTION = 158
EXP_2_HOUSE_20_75_EPOCHS_ANALYSIS_10: DESCRIPTION = 159
EXP_2_HOUSE_20_25_EPOCHS_ANALYSIS_20: DESCRIPTION = 160
EXP_2_HOUSE_20_50_EPOCHS_ANALYSIS_20: DESCRIPTION = 161
EXP_2_HOUSE_20_75_EPOCHS_ANALYSIS_20: DESCRIPTION = 162
EXP_2_HOUSE_20_25_EPOCHS_ANALYSIS_40: DESCRIPTION = 163
EXP_2_HOUSE_20_50_EPOCHS_ANALYSIS_40: DESCRIPTION = 164
EXP_2_HOUSE_20_75_EPOCHS_ANALYSIS_40: DESCRIPTION = 165
EXP_2_HOUSE_20_25_EPOCHS_ANALYSIS_60: DESCRIPTION = 166
EXP_2_HOUSE_20_50_EPOCHS_ANALYSIS_60: DESCRIPTION = 167
EXP_2_HOUSE_20_75_EPOCHS_ANALYSIS_60: DESCRIPTION = 168

EXP_1_HOUSE_21_EPOCHS_ANALYSIS_10: DESCRIPTION = 169
EXP_1_HOUSE_21_EPOCHS_ANALYSIS_20: DESCRIPTION = 170
EXP_1_HOUSE_21_EPOCHS_ANALYSIS_40: DESCRIPTION = 171
EXP_1_HOUSE_21_EPOCHS_ANALYSIS_60: DESCRIPTION = 172

EXP_2_HOUSE_21_25_EPOCHS_ANALYSIS_10: DESCRIPTION = 173
EXP_2_HOUSE_21_50_EPOCHS_ANALYSIS_10: DESCRIPTION = 174
EXP_2_HOUSE_21_75_EPOCHS_ANALYSIS_10: DESCRIPTION = 175
EXP_2_HOUSE_21_25_EPOCHS_ANALYSIS_20: DESCRIPTION = 176
EXP_2_HOUSE_21_50_EPOCHS_ANALYSIS_20: DESCRIPTION = 177
EXP_2_HOUSE_21_75_EPOCHS_ANALYSIS_20: DESCRIPTION = 178
EXP_2_HOUSE_21_25_EPOCHS_ANALYSIS_40: DESCRIPTION = 179
EXP_2_HOUSE_21_50_EPOCHS_ANALYSIS_40: DESCRIPTION = 180
EXP_2_HOUSE_21_75_EPOCHS_ANALYSIS_40: DESCRIPTION = 181
EXP_2_HOUSE_21_25_EPOCHS_ANALYSIS_60: DESCRIPTION = 182
EXP_2_HOUSE_21_50_EPOCHS_ANALYSIS_60: DESCRIPTION = 183
EXP_2_HOUSE_21_75_EPOCHS_ANALYSIS_60: DESCRIPTION = 184

EXP_1_HOUSE_22_EPOCHS_ANALYSIS_10: DESCRIPTION = 185
EXP_1_HOUSE_22_EPOCHS_ANALYSIS_20: DESCRIPTION = 186
EXP_1_HOUSE_22_EPOCHS_ANALYSIS_40: DESCRIPTION = 187
EXP_1_HOUSE_22_EPOCHS_ANALYSIS_60: DESCRIPTION = 188

EXP_2_HOUSE_22_25_EPOCHS_ANALYSIS_10: DESCRIPTION = 189
EXP_2_HOUSE_22_50_EPOCHS_ANALYSIS_10: DESCRIPTION = 190
EXP_2_HOUSE_22_75_EPOCHS_ANALYSIS_10: DESCRIPTION = 191
EXP_2_HOUSE_22_25_EPOCHS_ANALYSIS_20: DESCRIPTION = 192
EXP_2_HOUSE_22_50_EPOCHS_ANALYSIS_20: DESCRIPTION = 193
EXP_2_HOUSE_22_75_EPOCHS_ANALYSIS_20: DESCRIPTION = 194
EXP_2_HOUSE_22_25_EPOCHS_ANALYSIS_40: DESCRIPTION = 195
EXP_2_HOUSE_22_50_EPOCHS_ANALYSIS_40: DESCRIPTION = 196
EXP_2_HOUSE_22_75_EPOCHS_ANALYSIS_40: DESCRIPTION = 197
EXP_2_HOUSE_22_25_EPOCHS_ANALYSIS_60: DESCRIPTION = 198
EXP_2_HOUSE_22_50_EPOCHS_ANALYSIS_60: DESCRIPTION = 199
EXP_2_HOUSE_22_75_EPOCHS_ANALYSIS_60: DESCRIPTION = 200

EXP_1_HOUSE_1_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 201
EXP_1_HOUSE_1_2_LAYERS_BACKBONE_60_EPOCHS: DESCRIPTION = 202

EXP_2_HOUSE_1_25_40_GENERAL_2_LAYERS_BACKBONE_20_EPOCHS: DESCRIPTION = 203
EXP_2_HOUSE_1_25_60_GENERAL_2_LAYERS_BACKBONE_20_EPOCHS: DESCRIPTION = 204
EXP_2_HOUSE_1_25_40_GENERAL_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 205
EXP_2_HOUSE_1_25_60_GENERAL_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 206
EXP_2_HOUSE_1_50_40_GENERAL_2_LAYERS_BACKBONE_20_EPOCHS: DESCRIPTION = 207
EXP_2_HOUSE_1_50_60_GENERAL_2_LAYERS_BACKBONE_20_EPOCHS: DESCRIPTION = 208
EXP_2_HOUSE_1_50_40_GENERAL_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 209
EXP_2_HOUSE_1_50_60_GENERAL_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 210
EXP_2_HOUSE_1_75_40_GENERAL_2_LAYERS_BACKBONE_20_EPOCHS: DESCRIPTION = 211
EXP_2_HOUSE_1_75_60_GENERAL_2_LAYERS_BACKBONE_20_EPOCHS: DESCRIPTION = 212
EXP_2_HOUSE_1_75_40_GENERAL_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 213
EXP_2_HOUSE_1_75_60_GENERAL_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 214

EXP_1_HOUSE_2_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 215
EXP_1_HOUSE_2_2_LAYERS_BACKBONE_60_EPOCHS: DESCRIPTION = 216

EXP_2_HOUSE_2_25_40_GENERAL_2_LAYERS_BACKBONE_20_EPOCHS: DESCRIPTION = 217
EXP_2_HOUSE_2_25_60_GENERAL_2_LAYERS_BACKBONE_20_EPOCHS: DESCRIPTION = 218
EXP_2_HOUSE_2_25_40_GENERAL_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 219
EXP_2_HOUSE_2_25_60_GENERAL_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 220
EXP_2_HOUSE_2_50_40_GENERAL_2_LAYERS_BACKBONE_20_EPOCHS: DESCRIPTION = 221
EXP_2_HOUSE_2_50_60_GENERAL_2_LAYERS_BACKBONE_20_EPOCHS: DESCRIPTION = 222
EXP_2_HOUSE_2_50_40_GENERAL_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 223
EXP_2_HOUSE_2_50_60_GENERAL_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 224
EXP_2_HOUSE_2_75_40_GENERAL_2_LAYERS_BACKBONE_20_EPOCHS: DESCRIPTION = 225
EXP_2_HOUSE_2_75_60_GENERAL_2_LAYERS_BACKBONE_20_EPOCHS: DESCRIPTION = 226
EXP_2_HOUSE_2_75_40_GENERAL_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 227
EXP_2_HOUSE_2_75_60_GENERAL_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 228

EXP_1_HOUSE_7_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 229
EXP_1_HOUSE_7_2_LAYERS_BACKBONE_60_EPOCHS: DESCRIPTION = 230

EXP_2_HOUSE_7_25_40_GENERAL_2_LAYERS_BACKBONE_20_EPOCHS: DESCRIPTION = 231
EXP_2_HOUSE_7_25_60_GENERAL_2_LAYERS_BACKBONE_20_EPOCHS: DESCRIPTION = 232
EXP_2_HOUSE_7_25_40_GENERAL_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 233
EXP_2_HOUSE_7_25_60_GENERAL_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 234
EXP_2_HOUSE_7_50_40_GENERAL_2_LAYERS_BACKBONE_20_EPOCHS: DESCRIPTION = 235
EXP_2_HOUSE_7_50_60_GENERAL_2_LAYERS_BACKBONE_20_EPOCHS: DESCRIPTION = 236
EXP_2_HOUSE_7_50_40_GENERAL_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 237
EXP_2_HOUSE_7_50_60_GENERAL_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 238
EXP_2_HOUSE_7_75_40_GENERAL_2_LAYERS_BACKBONE_20_EPOCHS: DESCRIPTION = 239
EXP_2_HOUSE_7_75_60_GENERAL_2_LAYERS_BACKBONE_20_EPOCHS: DESCRIPTION = 240
EXP_2_HOUSE_7_75_40_GENERAL_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 241
EXP_2_HOUSE_7_75_60_GENERAL_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 242

EXP_1_HOUSE_9_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 243
EXP_1_HOUSE_9_2_LAYERS_BACKBONE_60_EPOCHS: DESCRIPTION = 244

EXP_2_HOUSE_9_25_40_GENERAL_2_LAYERS_BACKBONE_20_EPOCHS: DESCRIPTION = 245
EXP_2_HOUSE_9_25_60_GENERAL_2_LAYERS_BACKBONE_20_EPOCHS: DESCRIPTION = 246
EXP_2_HOUSE_9_25_40_GENERAL_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 247
EXP_2_HOUSE_9_25_60_GENERAL_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 248
EXP_2_HOUSE_9_50_40_GENERAL_2_LAYERS_BACKBONE_20_EPOCHS: DESCRIPTION = 249
EXP_2_HOUSE_9_50_60_GENERAL_2_LAYERS_BACKBONE_20_EPOCHS: DESCRIPTION = 250
EXP_2_HOUSE_9_50_40_GENERAL_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 251
EXP_2_HOUSE_9_50_60_GENERAL_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 252
EXP_2_HOUSE_9_75_40_GENERAL_2_LAYERS_BACKBONE_20_EPOCHS: DESCRIPTION = 253
EXP_2_HOUSE_9_75_60_GENERAL_2_LAYERS_BACKBONE_20_EPOCHS: DESCRIPTION = 254
EXP_2_HOUSE_9_75_40_GENERAL_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 255
EXP_2_HOUSE_9_75_60_GENERAL_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 256

EXP_1_HOUSE_10_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 257
EXP_1_HOUSE_10_2_LAYERS_BACKBONE_60_EPOCHS: DESCRIPTION = 258

EXP_2_HOUSE_10_25_40_GENERAL_2_LAYERS_BACKBONE_20_EPOCHS: DESCRIPTION = 259
EXP_2_HOUSE_10_25_60_GENERAL_2_LAYERS_BACKBONE_20_EPOCHS: DESCRIPTION = 260
EXP_2_HOUSE_10_25_40_GENERAL_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 261
EXP_2_HOUSE_10_25_60_GENERAL_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 262
EXP_2_HOUSE_10_50_40_GENERAL_2_LAYERS_BACKBONE_20_EPOCHS: DESCRIPTION = 263
EXP_2_HOUSE_10_50_60_GENERAL_2_LAYERS_BACKBONE_20_EPOCHS: DESCRIPTION = 264
EXP_2_HOUSE_10_50_40_GENERAL_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 265
EXP_2_HOUSE_10_50_60_GENERAL_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 266
EXP_2_HOUSE_10_75_40_GENERAL_2_LAYERS_BACKBONE_20_EPOCHS: DESCRIPTION = 267
EXP_2_HOUSE_10_75_60_GENERAL_2_LAYERS_BACKBONE_20_EPOCHS: DESCRIPTION = 268
EXP_2_HOUSE_10_75_40_GENERAL_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 269
EXP_2_HOUSE_10_75_60_GENERAL_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 270

EXP_1_HOUSE_13_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 271
EXP_1_HOUSE_13_2_LAYERS_BACKBONE_60_EPOCHS: DESCRIPTION = 272

EXP_2_HOUSE_13_25_40_GENERAL_2_LAYERS_BACKBONE_20_EPOCHS: DESCRIPTION = 273
EXP_2_HOUSE_13_25_60_GENERAL_2_LAYERS_BACKBONE_20_EPOCHS: DESCRIPTION = 274
EXP_2_HOUSE_13_25_40_GENERAL_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 275
EXP_2_HOUSE_13_25_60_GENERAL_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 276
EXP_2_HOUSE_13_50_40_GENERAL_2_LAYERS_BACKBONE_20_EPOCHS: DESCRIPTION = 277
EXP_2_HOUSE_13_50_60_GENERAL_2_LAYERS_BACKBONE_20_EPOCHS: DESCRIPTION = 278
EXP_2_HOUSE_13_50_40_GENERAL_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 279
EXP_2_HOUSE_13_50_60_GENERAL_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 280
EXP_2_HOUSE_13_75_40_GENERAL_2_LAYERS_BACKBONE_20_EPOCHS: DESCRIPTION = 281
EXP_2_HOUSE_13_75_60_GENERAL_2_LAYERS_BACKBONE_20_EPOCHS: DESCRIPTION = 282
EXP_2_HOUSE_13_75_40_GENERAL_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 283
EXP_2_HOUSE_13_75_60_GENERAL_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 284

EXP_1_HOUSE_15_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 285
EXP_1_HOUSE_15_2_LAYERS_BACKBONE_60_EPOCHS: DESCRIPTION = 286

EXP_2_HOUSE_15_25_40_GENERAL_2_LAYERS_BACKBONE_20_EPOCHS: DESCRIPTION = 287
EXP_2_HOUSE_15_25_60_GENERAL_2_LAYERS_BACKBONE_20_EPOCHS: DESCRIPTION = 288
EXP_2_HOUSE_15_25_40_GENERAL_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 289
EXP_2_HOUSE_15_25_60_GENERAL_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 290
EXP_2_HOUSE_15_50_40_GENERAL_2_LAYERS_BACKBONE_20_EPOCHS: DESCRIPTION = 291
EXP_2_HOUSE_15_50_60_GENERAL_2_LAYERS_BACKBONE_20_EPOCHS: DESCRIPTION = 292
EXP_2_HOUSE_15_50_40_GENERAL_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 293
EXP_2_HOUSE_15_50_60_GENERAL_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 294
EXP_2_HOUSE_15_75_40_GENERAL_2_LAYERS_BACKBONE_20_EPOCHS: DESCRIPTION = 295
EXP_2_HOUSE_15_75_60_GENERAL_2_LAYERS_BACKBONE_20_EPOCHS: DESCRIPTION = 296
EXP_2_HOUSE_15_75_40_GENERAL_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 297
EXP_2_HOUSE_15_75_60_GENERAL_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 298

EXP_1_HOUSE_20_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 299
EXP_1_HOUSE_20_2_LAYERS_BACKBONE_60_EPOCHS: DESCRIPTION = 300

EXP_2_HOUSE_20_25_40_GENERAL_2_LAYERS_BACKBONE_20_EPOCHS: DESCRIPTION = 301
EXP_2_HOUSE_20_25_60_GENERAL_2_LAYERS_BACKBONE_20_EPOCHS: DESCRIPTION = 302
EXP_2_HOUSE_20_25_40_GENERAL_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 303
EXP_2_HOUSE_20_25_60_GENERAL_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 304
EXP_2_HOUSE_20_50_40_GENERAL_2_LAYERS_BACKBONE_20_EPOCHS: DESCRIPTION = 305
EXP_2_HOUSE_20_50_60_GENERAL_2_LAYERS_BACKBONE_20_EPOCHS: DESCRIPTION = 306
EXP_2_HOUSE_20_50_40_GENERAL_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 307
EXP_2_HOUSE_20_50_60_GENERAL_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 308
EXP_2_HOUSE_20_75_40_GENERAL_2_LAYERS_BACKBONE_20_EPOCHS: DESCRIPTION = 309
EXP_2_HOUSE_20_75_60_GENERAL_2_LAYERS_BACKBONE_20_EPOCHS: DESCRIPTION = 310
EXP_2_HOUSE_20_75_40_GENERAL_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 311
EXP_2_HOUSE_20_75_60_GENERAL_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 312

EXP_1_HOUSE_21_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 313
EXP_1_HOUSE_21_2_LAYERS_BACKBONE_60_EPOCHS: DESCRIPTION = 314

EXP_2_HOUSE_21_25_40_GENERAL_2_LAYERS_BACKBONE_20_EPOCHS: DESCRIPTION = 315
EXP_2_HOUSE_21_25_60_GENERAL_2_LAYERS_BACKBONE_20_EPOCHS: DESCRIPTION = 316
EXP_2_HOUSE_21_25_40_GENERAL_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 317
EXP_2_HOUSE_21_25_60_GENERAL_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 318
EXP_2_HOUSE_21_50_40_GENERAL_2_LAYERS_BACKBONE_20_EPOCHS: DESCRIPTION = 319
EXP_2_HOUSE_21_50_60_GENERAL_2_LAYERS_BACKBONE_20_EPOCHS: DESCRIPTION = 320
EXP_2_HOUSE_21_50_40_GENERAL_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 321
EXP_2_HOUSE_21_50_60_GENERAL_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 322
EXP_2_HOUSE_21_75_40_GENERAL_2_LAYERS_BACKBONE_20_EPOCHS: DESCRIPTION = 323
EXP_2_HOUSE_21_75_60_GENERAL_2_LAYERS_BACKBONE_20_EPOCHS: DESCRIPTION = 324
EXP_2_HOUSE_21_75_40_GENERAL_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 325
EXP_2_HOUSE_21_75_60_GENERAL_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 326

EXP_1_HOUSE_22_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 327
EXP_1_HOUSE_22_2_LAYERS_BACKBONE_60_EPOCHS: DESCRIPTION = 328

EXP_2_HOUSE_22_25_40_GENERAL_2_LAYERS_BACKBONE_20_EPOCHS: DESCRIPTION = 329
EXP_2_HOUSE_22_25_60_GENERAL_2_LAYERS_BACKBONE_20_EPOCHS: DESCRIPTION = 330
EXP_2_HOUSE_22_25_40_GENERAL_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 331
EXP_2_HOUSE_22_25_60_GENERAL_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 332
EXP_2_HOUSE_22_50_40_GENERAL_2_LAYERS_BACKBONE_20_EPOCHS: DESCRIPTION = 333
EXP_2_HOUSE_22_50_60_GENERAL_2_LAYERS_BACKBONE_20_EPOCHS: DESCRIPTION = 334
EXP_2_HOUSE_22_50_40_GENERAL_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 335
EXP_2_HOUSE_22_50_60_GENERAL_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 336
EXP_2_HOUSE_22_75_40_GENERAL_2_LAYERS_BACKBONE_20_EPOCHS: DESCRIPTION = 337
EXP_2_HOUSE_22_75_60_GENERAL_2_LAYERS_BACKBONE_20_EPOCHS: DESCRIPTION = 338
EXP_2_HOUSE_22_75_40_GENERAL_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 339
EXP_2_HOUSE_22_75_60_GENERAL_2_LAYERS_BACKBONE_40_EPOCHS: DESCRIPTION = 340

EXP_1_HOUSE_1_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 361
EXP_1_HOUSE_1_FIXED_BACKBONE_60_EPOCHS: DESCRIPTION = 362

EXP_2_HOUSE_1_25_40_GENERAL_FIXED_BACKBONE_20_EPOCHS: DESCRIPTION = 363
EXP_2_HOUSE_1_25_60_GENERAL_FIXED_BACKBONE_20_EPOCHS: DESCRIPTION = 364
EXP_2_HOUSE_1_25_40_GENERAL_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 365
EXP_2_HOUSE_1_25_60_GENERAL_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 366
EXP_2_HOUSE_1_50_40_GENERAL_FIXED_BACKBONE_20_EPOCHS: DESCRIPTION = 367
EXP_2_HOUSE_1_50_60_GENERAL_FIXED_BACKBONE_20_EPOCHS: DESCRIPTION = 368
EXP_2_HOUSE_1_50_40_GENERAL_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 369
EXP_2_HOUSE_1_50_60_GENERAL_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 370
EXP_2_HOUSE_1_75_40_GENERAL_FIXED_BACKBONE_20_EPOCHS: DESCRIPTION = 371
EXP_2_HOUSE_1_75_60_GENERAL_FIXED_BACKBONE_20_EPOCHS: DESCRIPTION = 372
EXP_2_HOUSE_1_75_40_GENERAL_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 373
EXP_2_HOUSE_1_75_60_GENERAL_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 374

EXP_1_HOUSE_2_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 375
EXP_1_HOUSE_2_FIXED_BACKBONE_60_EPOCHS: DESCRIPTION = 376

EXP_2_HOUSE_2_25_40_GENERAL_FIXED_BACKBONE_20_EPOCHS: DESCRIPTION = 377
EXP_2_HOUSE_2_25_60_GENERAL_FIXED_BACKBONE_20_EPOCHS: DESCRIPTION = 378
EXP_2_HOUSE_2_25_40_GENERAL_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 379
EXP_2_HOUSE_2_25_60_GENERAL_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 380
EXP_2_HOUSE_2_50_40_GENERAL_FIXED_BACKBONE_20_EPOCHS: DESCRIPTION = 381
EXP_2_HOUSE_2_50_60_GENERAL_FIXED_BACKBONE_20_EPOCHS: DESCRIPTION = 382
EXP_2_HOUSE_2_50_40_GENERAL_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 383
EXP_2_HOUSE_2_50_60_GENERAL_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 384
EXP_2_HOUSE_2_75_40_GENERAL_FIXED_BACKBONE_20_EPOCHS: DESCRIPTION = 385
EXP_2_HOUSE_2_75_60_GENERAL_FIXED_BACKBONE_20_EPOCHS: DESCRIPTION = 386
EXP_2_HOUSE_2_75_40_GENERAL_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 387
EXP_2_HOUSE_2_75_60_GENERAL_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 388

EXP_1_HOUSE_7_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 389
EXP_1_HOUSE_7_FIXED_BACKBONE_60_EPOCHS: DESCRIPTION = 390

EXP_2_HOUSE_7_25_40_GENERAL_FIXED_BACKBONE_20_EPOCHS: DESCRIPTION = 391
EXP_2_HOUSE_7_25_60_GENERAL_FIXED_BACKBONE_20_EPOCHS: DESCRIPTION = 392
EXP_2_HOUSE_7_25_40_GENERAL_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 393
EXP_2_HOUSE_7_25_60_GENERAL_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 394
EXP_2_HOUSE_7_50_40_GENERAL_FIXED_BACKBONE_20_EPOCHS: DESCRIPTION = 395
EXP_2_HOUSE_7_50_60_GENERAL_FIXED_BACKBONE_20_EPOCHS: DESCRIPTION = 396
EXP_2_HOUSE_7_50_40_GENERAL_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 397
EXP_2_HOUSE_7_50_60_GENERAL_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 398
EXP_2_HOUSE_7_75_40_GENERAL_FIXED_BACKBONE_20_EPOCHS: DESCRIPTION = 399
EXP_2_HOUSE_7_75_60_GENERAL_FIXED_BACKBONE_20_EPOCHS: DESCRIPTION = 400
EXP_2_HOUSE_7_75_40_GENERAL_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 401
EXP_2_HOUSE_7_75_60_GENERAL_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 402

EXP_1_HOUSE_9_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 403
EXP_1_HOUSE_9_FIXED_BACKBONE_60_EPOCHS: DESCRIPTION = 404

EXP_2_HOUSE_9_25_40_GENERAL_FIXED_BACKBONE_20_EPOCHS: DESCRIPTION = 405
EXP_2_HOUSE_9_25_60_GENERAL_FIXED_BACKBONE_20_EPOCHS: DESCRIPTION = 406
EXP_2_HOUSE_9_25_40_GENERAL_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 407
EXP_2_HOUSE_9_25_60_GENERAL_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 408
EXP_2_HOUSE_9_50_40_GENERAL_FIXED_BACKBONE_20_EPOCHS: DESCRIPTION = 409
EXP_2_HOUSE_9_50_60_GENERAL_FIXED_BACKBONE_20_EPOCHS: DESCRIPTION = 410
EXP_2_HOUSE_9_50_40_GENERAL_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 411
EXP_2_HOUSE_9_50_60_GENERAL_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 412
EXP_2_HOUSE_9_75_40_GENERAL_FIXED_BACKBONE_20_EPOCHS: DESCRIPTION = 413
EXP_2_HOUSE_9_75_60_GENERAL_FIXED_BACKBONE_20_EPOCHS: DESCRIPTION = 414
EXP_2_HOUSE_9_75_40_GENERAL_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 415
EXP_2_HOUSE_9_75_60_GENERAL_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 416

EXP_1_HOUSE_10_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 417
EXP_1_HOUSE_10_FIXED_BACKBONE_60_EPOCHS: DESCRIPTION = 418

EXP_2_HOUSE_10_25_40_GENERAL_FIXED_BACKBONE_20_EPOCHS: DESCRIPTION = 419
EXP_2_HOUSE_10_25_60_GENERAL_FIXED_BACKBONE_20_EPOCHS: DESCRIPTION = 420
EXP_2_HOUSE_10_25_40_GENERAL_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 421
EXP_2_HOUSE_10_25_60_GENERAL_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 422
EXP_2_HOUSE_10_50_40_GENERAL_FIXED_BACKBONE_20_EPOCHS: DESCRIPTION = 423
EXP_2_HOUSE_10_50_60_GENERAL_FIXED_BACKBONE_20_EPOCHS: DESCRIPTION = 424
EXP_2_HOUSE_10_50_40_GENERAL_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 425
EXP_2_HOUSE_10_50_60_GENERAL_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 426
EXP_2_HOUSE_10_75_40_GENERAL_FIXED_BACKBONE_20_EPOCHS: DESCRIPTION = 427
EXP_2_HOUSE_10_75_60_GENERAL_FIXED_BACKBONE_20_EPOCHS: DESCRIPTION = 428
EXP_2_HOUSE_10_75_40_GENERAL_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 429
EXP_2_HOUSE_10_75_60_GENERAL_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 430

EXP_1_HOUSE_13_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 431
EXP_1_HOUSE_13_FIXED_BACKBONE_60_EPOCHS: DESCRIPTION = 432

EXP_2_HOUSE_13_25_40_GENERAL_FIXED_BACKBONE_20_EPOCHS: DESCRIPTION = 433
EXP_2_HOUSE_13_25_60_GENERAL_FIXED_BACKBONE_20_EPOCHS: DESCRIPTION = 434
EXP_2_HOUSE_13_25_40_GENERAL_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 435
EXP_2_HOUSE_13_25_60_GENERAL_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 436
EXP_2_HOUSE_13_50_40_GENERAL_FIXED_BACKBONE_20_EPOCHS: DESCRIPTION = 437
EXP_2_HOUSE_13_50_60_GENERAL_FIXED_BACKBONE_20_EPOCHS: DESCRIPTION = 438
EXP_2_HOUSE_13_50_40_GENERAL_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 439
EXP_2_HOUSE_13_50_60_GENERAL_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 440
EXP_2_HOUSE_13_75_40_GENERAL_FIXED_BACKBONE_20_EPOCHS: DESCRIPTION = 441
EXP_2_HOUSE_13_75_60_GENERAL_FIXED_BACKBONE_20_EPOCHS: DESCRIPTION = 442
EXP_2_HOUSE_13_75_40_GENERAL_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 443
EXP_2_HOUSE_13_75_60_GENERAL_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 444

EXP_1_HOUSE_15_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 445
EXP_1_HOUSE_15_FIXED_BACKBONE_60_EPOCHS: DESCRIPTION = 446

EXP_2_HOUSE_15_25_40_GENERAL_FIXED_BACKBONE_20_EPOCHS: DESCRIPTION = 447
EXP_2_HOUSE_15_25_60_GENERAL_FIXED_BACKBONE_20_EPOCHS: DESCRIPTION = 448
EXP_2_HOUSE_15_25_40_GENERAL_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 449
EXP_2_HOUSE_15_25_60_GENERAL_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 450
EXP_2_HOUSE_15_50_40_GENERAL_FIXED_BACKBONE_20_EPOCHS: DESCRIPTION = 451
EXP_2_HOUSE_15_50_60_GENERAL_FIXED_BACKBONE_20_EPOCHS: DESCRIPTION = 452
EXP_2_HOUSE_15_50_40_GENERAL_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 453
EXP_2_HOUSE_15_50_60_GENERAL_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 454
EXP_2_HOUSE_15_75_40_GENERAL_FIXED_BACKBONE_20_EPOCHS: DESCRIPTION = 455
EXP_2_HOUSE_15_75_60_GENERAL_FIXED_BACKBONE_20_EPOCHS: DESCRIPTION = 456
EXP_2_HOUSE_15_75_40_GENERAL_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 457
EXP_2_HOUSE_15_75_60_GENERAL_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 458

EXP_1_HOUSE_20_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 459
EXP_1_HOUSE_20_FIXED_BACKBONE_60_EPOCHS: DESCRIPTION = 460

EXP_2_HOUSE_20_25_40_GENERAL_FIXED_BACKBONE_20_EPOCHS: DESCRIPTION = 461
EXP_2_HOUSE_20_25_60_GENERAL_FIXED_BACKBONE_20_EPOCHS: DESCRIPTION = 462
EXP_2_HOUSE_20_25_40_GENERAL_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 463
EXP_2_HOUSE_20_25_60_GENERAL_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 464
EXP_2_HOUSE_20_50_40_GENERAL_FIXED_BACKBONE_20_EPOCHS: DESCRIPTION = 465
EXP_2_HOUSE_20_50_60_GENERAL_FIXED_BACKBONE_20_EPOCHS: DESCRIPTION = 466
EXP_2_HOUSE_20_50_40_GENERAL_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 467
EXP_2_HOUSE_20_50_60_GENERAL_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 468
EXP_2_HOUSE_20_75_40_GENERAL_FIXED_BACKBONE_20_EPOCHS: DESCRIPTION = 469
EXP_2_HOUSE_20_75_60_GENERAL_FIXED_BACKBONE_20_EPOCHS: DESCRIPTION = 470
EXP_2_HOUSE_20_75_40_GENERAL_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 471
EXP_2_HOUSE_20_75_60_GENERAL_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 472

EXP_1_HOUSE_21_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 473
EXP_1_HOUSE_21_FIXED_BACKBONE_60_EPOCHS: DESCRIPTION = 474

EXP_2_HOUSE_21_25_40_GENERAL_FIXED_BACKBONE_20_EPOCHS: DESCRIPTION = 475
EXP_2_HOUSE_21_25_60_GENERAL_FIXED_BACKBONE_20_EPOCHS: DESCRIPTION = 476
EXP_2_HOUSE_21_25_40_GENERAL_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 477
EXP_2_HOUSE_21_25_60_GENERAL_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 478
EXP_2_HOUSE_21_50_40_GENERAL_FIXED_BACKBONE_20_EPOCHS: DESCRIPTION = 479
EXP_2_HOUSE_21_50_60_GENERAL_FIXED_BACKBONE_20_EPOCHS: DESCRIPTION = 480
EXP_2_HOUSE_21_50_40_GENERAL_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 481
EXP_2_HOUSE_21_50_60_GENERAL_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 482
EXP_2_HOUSE_21_75_40_GENERAL_FIXED_BACKBONE_20_EPOCHS: DESCRIPTION = 483
EXP_2_HOUSE_21_75_60_GENERAL_FIXED_BACKBONE_20_EPOCHS: DESCRIPTION = 484
EXP_2_HOUSE_21_75_40_GENERAL_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 485
EXP_2_HOUSE_21_75_60_GENERAL_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 486

EXP_1_HOUSE_22_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 487
EXP_1_HOUSE_22_FIXED_BACKBONE_60_EPOCHS: DESCRIPTION = 488

EXP_2_HOUSE_22_25_40_GENERAL_FIXED_BACKBONE_20_EPOCHS: DESCRIPTION = 489
EXP_2_HOUSE_22_25_60_GENERAL_FIXED_BACKBONE_20_EPOCHS: DESCRIPTION = 490
EXP_2_HOUSE_22_25_40_GENERAL_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 491
EXP_2_HOUSE_22_25_60_GENERAL_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 492
EXP_2_HOUSE_22_50_40_GENERAL_FIXED_BACKBONE_20_EPOCHS: DESCRIPTION = 493
EXP_2_HOUSE_22_50_60_GENERAL_FIXED_BACKBONE_20_EPOCHS: DESCRIPTION = 494
EXP_2_HOUSE_22_50_40_GENERAL_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 495
EXP_2_HOUSE_22_50_60_GENERAL_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 496
EXP_2_HOUSE_22_75_40_GENERAL_FIXED_BACKBONE_20_EPOCHS: DESCRIPTION = 497
EXP_2_HOUSE_22_75_60_GENERAL_FIXED_BACKBONE_20_EPOCHS: DESCRIPTION = 498
EXP_2_HOUSE_22_75_40_GENERAL_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 499
EXP_2_HOUSE_22_75_60_GENERAL_FIXED_BACKBONE_40_EPOCHS: DESCRIPTION = 500


class DetrDoorDetector(nn.Module):
    """
    This class builds a door detector starting from a detr pretrained module.
    Basically it loads a dtr module and modify its structure to recognize door.
    """
    def __init__(self, model_name: ModelName, n_labels: int, pretrained: bool, dataset_name: DATASET, description: DESCRIPTION):
        """

        :param model_name: the name of the detr base model
        :param n_labels: the labels' number
        :param pretrained: it refers to the DetrDoorDetector class, not to detr base model.
                            It True, the DetrDoorDetector's weights are loaded, otherwise the weights are loaded only for the detr base model
        """
        super(DetrDoorDetector, self).__init__()
        self._model_name = model_name
        self.model = torch.hub.load('facebookresearch/detr', model_name, pretrained=True)
        self._dataset_name = dataset_name
        self._description = description

        # Change the last part of the model
        self.model.query_embed = nn.Embedding(10, self.model.transformer.d_model)
        self.model.class_embed = nn.Linear(256, n_labels + 1)

        if pretrained:
            path = os.path.join('train_params', self._model_name + '_' + str(self._description), str(self._dataset_name))
            if trained_models_path == "":
                path = os.path.join(os.path.dirname(__file__), path)
            else:
                path = os.path.join(trained_models_path, path)
            self.model.load_state_dict(torch.load(os.path.join(path, 'model.pth')))

    def forward(self, x):
        x = self.model(x)

        """
        It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape=[batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the un-normalized bounding box.
               - "aux_outputs": Optional, only returned when auxiliary losses are activated. It is a list of
                                dictionaries containing the two above keys for each decoder layer.
        """
        return x

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    def to(self, device):
        self.model.to(device)

    def save(self, epoch, optimizer_state_dict, lr_scheduler_state_dict, params, logs):
        path = os.path.join('train_params', self._model_name + '_' + str(self._description), str(self._dataset_name))
        if trained_models_path == "":
            path = os.path.join(os.path.dirname(__file__), path)
        else:
            path = os.path.join(trained_models_path, path)

        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(self.model.state_dict(), os.path.join(path, 'model.pth'))
        torch.save(
            {
                'optimizer_state_dict': optimizer_state_dict,
                'lr_scheduler_state_dict': lr_scheduler_state_dict
            }, os.path.join(path, 'checkpoint.pth')
        )

        torch.save(
            {
                'epoch': epoch,
                'logs': logs,
                'params': params,
            }, os.path.join(path, 'training_data.pth')
        )

    def load_checkpoint(self,):
        path = os.path.join('train_params', self._model_name + '_' + str(self._description), str(self._dataset_name))
        if trained_models_path == "":
            path = os.path.join(os.path.dirname(__file__), path)
        else:
            path = os.path.join(trained_models_path, path)

        checkpoint = torch.load(os.path.join(path, 'checkpoint.pth'))
        training_data = torch.load(os.path.join(path, 'training_data.pth'))

        return {**checkpoint, **training_data}

    def set_description(self, description: DESCRIPTION):
        self._description = description



