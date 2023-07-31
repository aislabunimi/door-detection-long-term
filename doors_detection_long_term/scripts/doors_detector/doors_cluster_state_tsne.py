import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.pyplot import subplots, show
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm

from doors_detection_long_term.doors_detector.dataset.torch_dataset import DEEP_DOORS_2_LABELLED, FINAL_DOORS_DATASET
from doors_detection_long_term.doors_detector.models.detr import PostProcess
from doors_detection_long_term.doors_detector.models.detr_door_detector import *
from doors_detection_long_term.doors_detector.models.model_names import DETR_RESNET50
from doors_detection_long_term.doors_detector.utilities.collate_fn_functions import seed_everything, collate_fn
from dataset_configurator import get_deep_doors_2_labelled_sets, get_final_doors_dataset

device = 'cuda'
seed_everything(0)
batch_size = 1
values = {'transformer': [], 'max_scores': [], 'labels': []}

#train, test, labels, COLORS = get_final_doors_dataset(experiment=1, folder_name='house1', train_size=0.2, use_negatives=False)
train, test, labels, COLORS = get_deep_doors_2_labelled_sets()
model = DetrDoorDetector(model_name=DETR_RESNET50, n_labels=len(labels.keys()), pretrained=True, dataset_name=DEEP_DOORS_2_LABELLED, description=DEEP_DOORS_2_LABELLED_EXP)


def extract_tranformer_weights(model, input, output):
    tensor = output[0].detach()
    b = torch.split(tensor[-1], 1, 0)
    values['transformer'].extend(b)


model.model.transformer.register_forward_hook(
    extract_tranformer_weights
)

model.to(device)
model.eval()

data_loader_training = DataLoader(train, batch_size=batch_size, collate_fn=collate_fn, shuffle=False, num_workers=6)

post_processor = PostProcess()

for i, training_data in tqdm(enumerate(data_loader_training), total=len(data_loader_training)):
    images, targets = training_data

    images = images.to(device)
    outputs = model(images)

    processed_data = post_processor(outputs=outputs, target_sizes=torch.tensor([[100, 100] for _ in range(len(images))], device=device))
    for item in processed_data:
        scores = item['scores']
        max_score = torch.argmax(scores).item()
        values['max_scores'].append(max_score)
        values['labels'].append(item['labels'][max_score].item())

flatten_encoder = np.array([torch.squeeze(v)[m].flatten().tolist() for v, m in zip(values['transformer'], values['max_scores'])])
flatten_encoder_pca_train = PCA(n_components=50, random_state=42).fit_transform(flatten_encoder)
color_list = np.array([COLORS[k] for k in sorted(COLORS.keys())])
color_train = color_list[values['labels']]

values = {'transformer': [], 'max_scores': [], 'labels': []}

# TEST
data_loader_test = DataLoader(test, batch_size=batch_size, collate_fn=collate_fn, shuffle=False, num_workers=6)

post_processor = PostProcess()

for i, training_data in tqdm(enumerate(data_loader_test), total=len(data_loader_test)):
    images, targets = training_data

    images = images.to(device)
    outputs = model(images)

    processed_data = post_processor(outputs=outputs, target_sizes=torch.tensor([[100, 100] for _ in range(len(images))], device=device))
    for item in processed_data:
        scores = item['scores']
        max_score = torch.argmax(scores).item()
        values['max_scores'].append(max_score)
        values['labels'].append(item['labels'][max_score].item())

flatten_encoder = np.array([torch.squeeze(v)[m].flatten().tolist() for v, m in zip(values['transformer'], values['max_scores'])])
flatten_encoder_pca_test = PCA(n_components=50, random_state=42).fit_transform(flatten_encoder)

fig, axes = subplots(nrows=1, ncols=2, figsize=(8, 4))
color_test = color_list[values['labels']]

for title, flatten, colors, axis in tqdm(zip(['Train set', 'Test set'], [flatten_encoder_pca_train, flatten_encoder_pca_test], [color_train, color_test], axes.flatten()), desc="Computing TSNEs", total=1):
    axis.scatter(*TSNE(n_components=2, perplexity=100).fit_transform(flatten).T, s=1, c=colors)
    axis.xaxis.set_visible(False)
    axis.yaxis.set_visible(False)
    axis.set_title("TSNE - " + title, fontdict={'fontsize': 10,
                                                                                'fontweight': 10,
                                                                                'verticalalignment': 'baseline',
                                                                                'horizontalalignment': 'center'})

legend_elements = [Line2D([0], [0], marker='o', color='w', label=labels[l],
                           markerfacecolor=COLORS[l], markersize=8) for l in sorted(labels.keys())]

fig.legend(handles=legend_elements, loc='lower center', ncol=len(labels.keys()))

plt.show()
