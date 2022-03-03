# Enhancing Door Detection for Autonomous Mobile Robots with Environment-Specific Data Collection

Here you can find the code and the datasets used in the article entitled **Enhancing Door Detection for Autonomous Mobile Robots with Environment-Specific Data Collection**
To use this package and install all the dependencies, clone this repository and run:

```pip install -e .```

Datasets links:
* The relabelled version of *DeepDoors2* dataset [link](https://drive.google.com/file/d/1wSmFUHF9aSJkomwFdOmepMevBvkRpf3D/view?usp=sharing)
* The collected dataset [link](https://drive.google.com/file/d/1BqjBpobjKTomFjDkzhWjmCryAXOEluO2/view?usp=sharing)

## Simulation Environment

To acquire the visual dataset we use an extended version of Gibson, obtainable [here](https://github.com/micheleantonazzi/GibsonEnv.git).
The simulator is automatically installed with the above command `pip install -e .`.
The door dataset has been acquired by virtualizing through Gibson the environments of Matterport3D. 



## Pose extractor

The code to extract plausible positions of a mobile robot to acquire the images is in [positions_extractor](doors_detection_long_term/positions_extractor) package.

## Baseline
The code of the baseline (with the relative configuration parameters) is in [baseline.py](doors_detection_long_term/doors_detector/baseline/baseline.py).

## The door detector

The proposed detector is coded [here](doors_detection_long_term/doors_detector/models).

