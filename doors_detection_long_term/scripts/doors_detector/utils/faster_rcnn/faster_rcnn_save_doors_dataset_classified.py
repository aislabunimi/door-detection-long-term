import cv2
from torch.utils.data import DataLoader
from tqdm import tqdm

from doors_detection_long_term.doors_detector.dataset.torch_dataset import FINAL_DOORS_DATASET
from doors_detection_long_term.doors_detector.models.faster_rcnn import *
from doors_detection_long_term.doors_detector.models.model_names import FASTER_RCNN
from doors_detection_long_term.doors_detector.utilities.collate_fn_functions import collate_fn_faster_rcnn
from doors_detection_long_term.scripts.doors_detector.dataset_configurator import *
import torchvision.transforms as T

save_path = '/home/antonazzi/Downloads/test_faster_rcnn'

if not os.path.exists(save_path):
    os.mkdir(save_path)
#train, test, _, _ = get_final_doors_dataset_all_envs()
#train, test, labels, _ = get_final_doors_dataset_real_data(folder_name='floor1', train_size=0.25)
train, validation, test, labels, COLORS = get_final_doors_dataset_epoch_analysis(experiment=1, train_size=0.25, folder_name='house20')
data_loader_test = DataLoader(test, batch_size=1, collate_fn=collate_fn_faster_rcnn, drop_last=False, num_workers=4)

model = FasterRCNN(model_name=FASTER_RCNN, n_labels=3, pretrained=True, dataset_name=FINAL_DOORS_DATASET, description=EXP_1_HOUSE_20_60_EPOCHS)
model.eval()
model.to('cuda')
padding_height = 0
padding_width = 0
transform = T.Compose([
    T.Pad([padding_width, padding_height]) # Check according image size
])
with torch.no_grad():
    for i in range(len(test)):
        img, target, door_sample = test[i]
        img = img.unsqueeze(0).to('cuda')
        outputs = model.model(img)
        #print(outputs)
        img_size = list(img.size()[2:])
        for image, output in zip(img, outputs):
            output = apply_nms(output, confidence_threshold=0.75)
            save_image =door_sample.get_bgr_image().copy()
            for [x1, y1, x2, y2], label, conf in zip(output['boxes'], output['labels'], output['scores']):
                label, x1, y1, x2, y2 = label.item(), x1.item(), y1.item(), x2.item(), y2.item()
                label -= 1
                x1 = int(min(img_size[1], max(.0, x1)))
                y1 = int(min(img_size[0], max(.0, y1)))
                x2 = int(min(img_size[1], max(.0, x2)))
                y2 = int(min(img_size[0], max(.0, y2)))
                colors = {0: (0, 0, 255), 1: (0, 255, 0)}
                save_image = cv2.rectangle(save_image, (x1, y1), (x2, y2), colors[label], 2)
            cv2.imwrite(os.path.join(save_path, 'image_{0:05d}.png'.format(i)), save_image)
