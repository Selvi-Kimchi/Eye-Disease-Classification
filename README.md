# Eye Disease Image Classification
## Overview
This repository contains the code and resources for an eye disease image classification project using convolutional neural networks (CNNs). The project aims to accurately classify images of eyes into one of several categories, potentially identifying different eye diseases.

## Models
Two different models are implemented in this project:
* Custom CNN Model: A custom convolutional neural network designed and trained from scratch.
* DenseNet121 Model: A pre-trained DenseNet121 model fine-tuned for the eye disease classification task.
  
**Custom CNN Model**

* The custom CNN model is built using the following architecture:
* Conv2D layers with increasing filters and kernel sizes of (3, 3)
* MaxPooling2D layers for down-sampling
* Flatten layer to convert the 2D matrix to a vector
* Dense layers for classification, ending with a softmax activation function for multi-class classification

**DenseNet121 Model**

DenseNet121 is a popular and powerful CNN architecture that connects each layer to every other layer in a feed-forward fashion. The model is fine-tuned on the eye disease dataset to leverage the pre-trained weights on ImageNet and improve classification accuracy.

## Repository Structure
* data/: Contains the dataset of eye images.
* notebooks/: Jupyter notebooks for data exploration and model training.
* src/: Source code for model definitions and training scripts.
* results/: Directory to save model weights and training logs.

## How to Use
* Install the required dependencies.
* Run the training scripts to train the models.
* Use the trained models for classification on new eye images.
* 
## Dependencies
* TensorFlow
* Keras
* NumPy
* Matplotlib
