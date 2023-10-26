# Incipient Slip Detection PapillArray
Official implementation of paper: [Robust Learning-Based Incipient Slip Detection using the PapillArray Optical Tactile Sensor for Improved Robotic Gripping](https://arxiv.org/pdf/2307.04011.pdf)

## Installation
**1.** Our implementations are tested only on [Anadonda](https://www.anaconda.com/products/distribution). We recommend you install Anaconda if you would reproduce our work, although other development environments should also work. Our python version is 3.8.13

**2.** Activate your vitural environment(or others) and run:

    pip install -r requirements.txt

## Data
You should download our dataset from [THIS LINK](https://drive.google.com/drive/folders/1wGuRzLHXnhaB8dtsyesGwO4MZjJLhYRW?usp=drive_link) and ensure that you specify the local location of these datasets in the script's arguments. Alternatively, you need to create a folder named "datasets" at the root of your project and organize the data as follows:
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

## Pre-trained models
We also provide pretrained models in the **pre-trained-models** directory of this repository.
