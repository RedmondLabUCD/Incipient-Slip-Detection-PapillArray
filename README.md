# Incipient Slip Detection PapillArray
Official implementation of paper: [Robust Learning-Based Incipient Slip Detection using the PapillArray Optical Tactile Sensor for Improved Robotic Gripping](https://arxiv.org/pdf/2307.04011.pdf)

## Installation
**1.** Our implementations are tested only on [Anadonda](https://www.anaconda.com/products/distribution). We recommend you install Anaconda if you would reproduce our work, although other development environments should also work. Our python version is 3.8.13

**2.** Activate your vitural environment(or others) and run:

    pip install -r requirements.txt

## Data
You should download our dataset from [THIS LINK](https://zenodo.org/records/13228084?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImM2NTE2MGZhLWM1ZWMtNDVjYi1hNmU5LWFmOWY3YjRmODQ3MyIsImRhdGEiOnt9LCJyYW5kb20iOiI3Yjg5YTUwYjM4MWYwNWJkOTUxN2YwMTE5MmI0NjIwZSJ9.QcjCzNmS4VQ0eVNRXgMfauVk0R5ggBvWf5HAdT8twfhAtYrb5k4Y1DmqMBu0h5MnRq_V3jBpyoxJCdNFPTniSA) and ensure that you specify the local location of these datasets in the script's arguments. Alternatively, you need to create a folder named "datasets" at the root of your project and organize the data as follows:
```
.
├── datasets/
│   ├── merge/
│   │   └── dataset_train.npy
│   └── pillar_data/
│       ├── pillar_data_train.npy
│       └── pillar_data_test.npy
```
It's particularly worth noting that the data in **merge/dataset_train.npy** has already undergone data augmentation and is ready for neural network training. This contains a dataset prepared for the final deployment model. Based on our tests, training directly with this data does not result in overfitting. If you need to split the data into training and test sets, you can use the files in **pillar_data** and the scripts in **DATA_xx** to process.

For evaluating the model using all the **gripping data** collected from the gripper, you'll need to download it locally and then specify the exact path to the data. We've marked the sections in the script where edits are needed and provided examples for reference.

## Pre-trained models
We also provide pretrained models in the **pre-trained-models** directory of this repository.

## Cite this work
```
@article{incipient-slip-papillarray,
  title={Robust Learning-Based Incipient Slip Detection using the PapillArray Optical Tactile Sensor for Improved Robotic Gripping},
  author={Wang, Qiang and Ulloa, Pablo Martinez and Burke, Robert and Bulens, David Cordova and Redmond, Stephen J},
  journal={arXiv preprint arXiv:2307.04011},
  year={2023}
}
```
