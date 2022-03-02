# Enhancing Door Detection for Autonomous Mobile Robots with Environment-Specific Data Collection

Here you can find the code and the datasets used in the article entitled **Enhancing Door Detection for Autonomous Mobile Robots with Environment-Specific Data Collection**.
To use this package and install all the dependencies, clone this repository and run `pip install -e .`.


## Simulation Environment

To acquire the visual dataset we use an extended version of Gibson, obtainable [here](https://github.com/micheleantonazzi/GibsonEnv.git).
The simulator is automatically installed with the above command `pip install -e .`.
The door dataset has been acquired by virtualizing through Gibson the environments of Matterport3D. 

* The relabelled version of *DeepDoors2* dataset [link](https://drive.google.com/file/d/1wSmFUHF9aSJkomwFdOmepMevBvkRpf3D/view?usp=sharing)
* The collected dataset [link](https://drive.google.com/file/d/1BqjBpobjKTomFjDkzhWjmCryAXOEluO2/view?usp=sharing)
* 
## Pose extractor

The code to extract plausible positions of a mobile robot to acquire the images is in []

* The code of the proposed door detector and the baseline [link](https://github.com/micheleantonazzi/master-thesis-robust-door-detector)
* The proposed simulation environment [link](https://github.com/micheleantonazzi/GibsonEnv.git)
* The code to determine a set of relevant poses for a mobile robot in an environment [link](https://github.com/micheleantonazzi/gibson-env-utilities) 

