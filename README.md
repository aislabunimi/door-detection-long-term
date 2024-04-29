# Development and Adaptation of Robotic Vision in the Real–World: the Challenge of Door Detection
```
@misc{antonazzi2024development,
      title={Development and Adaptation of Robotic Vision in the Real-World: the Challenge of Door Detection}, 
      author={Michele Antonazzi and Matteo Luperto and N. Alberto Borghese and Nicola Basilico},
      year={2024},
}
```
Here you can find the code and the datasets of the article entitled **Development and Adaptation of Robotic Vision in the Real–World: the Challenge of Door Detection** [link](
https://doi.org/10.48550/arXiv.2401.17996)

Installation: `pip3 install -e .`

Examples inferece code:
* YOLOv5 [here](doors_detection_long_term/scripts/doors_detector/utils/yolov5/yolov5_classify_folder_and_save_samples.py)
* Faster R-CNN [here](doors_detection_long_term/scripts/doors_detector/utils/faster_rcnn/faster_rcnn_save_folder_classified.py)
* DETR [here](doors_detection_long_term/scripts/doors_detector/utils/detr/detr_classify_folder_and_save_samples.py)


Code:
* The code related to the position extraction algorithm is contained inside [positions_extractor](doors_detection_long_term/positions_extractor) package.
* The source code of the object detectors (DETR, YOLOv5, and Faster R-CNN) used in the experimental campaign can be found in the [here](doors_detection_long_term/doors_detector/models)
* The scripts to perform the training and the evaluation of the models are in the [train_models](doors_detection_long_term/scripts/doors_detector/train_models) and [calculate results](doors_detection_long_term/scripts/doors_detector/calculate_results) packages respectively.

Usefull links:
* The *DeepDoors2* dataset [link](https://unimi2013-my.sharepoint.com/:u:/g/personal/michele_antonazzi_unimi_it/EX1sP643dctKswmWsVBiCLoBtyOdkSsxi5fpZJy3mnoaBg?e=tGyuLU)
* The simulated dataset collected with iGibson [link](https://unimi2013-my.sharepoint.com/:f:/g/personal/michele_antonazzi_unimi_it/EppCQg0MemBIrNUYzww0dIsBvv9hiaBYwU61Gz6fB2kO8Q?e=pmHmcX)
* The photorealistic simulated dataset collected with Gibson [link](https://unimi2013-my.sharepoint.com/:u:/g/personal/michele_antonazzi_unimi_it/EVYqJ4lErGNIhzUpqK7HDjQBoz2vQ-17acmi3NCpmE2xRw?e=60PiZW)
* The real dataset acquired with the Giraff-X robotic platform in 4 real environments [link](https://unimi2013-my.sharepoint.com/:u:/g/personal/michele_antonazzi_unimi_it/EXLStATEcj9Hhd06k4AcU0EBTB7J3pUUG_At9Ar60NpI3g?e=xmEwP7)
* The parameters of the trained models can be downloaded [here](https://unimi2013-my.sharepoint.com/:f:/g/personal/michele_antonazzi_unimi_it/Er7n154eKXtHqESgk2MahoQBa_t7hka5grS7N4ELkamqvg?e=fGKzMF)


![](images/example.gif)  



