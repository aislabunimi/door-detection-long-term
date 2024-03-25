# R2SNet: Scalable Domain Adaptation for Object Detection in Cloud-Based Robots Ecosystems via Proposal Refinement

Here you can find the code and the datasets of the article entitled **R2SNet: Scalable Domain Adaptation for Object Detection in Cloud-Based Robots Ecosystems via Proposal Refinement**

Usefull links:
* The *DeepDoors2* dataset [link](https://unimi2013-my.sharepoint.com/:u:/g/personal/michele_antonazzi_unimi_it/EX1sP643dctKswmWsVBiCLoBtyOdkSsxi5fpZJy3mnoaBg?e=tGyuLU)
* The photorealistic simulated dataset collected with Gibson [link](https://unimi2013-my.sharepoint.com/:u:/g/personal/michele_antonazzi_unimi_it/EVYqJ4lErGNIhzUpqK7HDjQBoz2vQ-17acmi3NCpmE2xRw?e=60PiZW)
* The real dataset acquired with the Giraff-X robotic platform in 4 real environments [link](https://unimi2013-my.sharepoint.com/:u:/g/personal/michele_antonazzi_unimi_it/EXLStATEcj9Hhd06k4AcU0EBTB7J3pUUG_At9Ar60NpI3g?e=xmEwP7)
* The parameters of the trained models can be downloaded [here](https://unimi2013-my.sharepoint.com/:f:/g/personal/michele_antonazzi_unimi_it/Er7n154eKXtHqESgk2MahoQBa_t7hka5grS7N4ELkamqvg?e=fGKzMF)

Code:
* The R2SNet [implementation](doors_detection_long_term/doors_detector/models/bbox_filter_network_geometric.py).
* The Background Feature extractor Network [architecture](doors_detection_long_term/doors_detector/models/background_grid_network.py).
* The scripts to perform the training are in the [train_models](doors_detection_long_term/scripts/doors_detector/train_models/) package.

![](images/r2snet_experiments.gif)  



