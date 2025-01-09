# CNN Model Training with CIFAR-10 Dataset

This project demonstrates the training of a Convolutional Neural Network (CNN) on the CIFAR-10 dataset using TensorFlow and Keras. The notebook includes data preprocessing, model definition, training, and evaluation.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Techniques Used](#techniques-used)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The goal of this project is to build and train a CNN model to classify images from the CIFAR-10 dataset into one of ten categories. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class.

## Features

- Data loading and preprocessing
- CNN model definition with multiple convolutional layers
- Model compilation and training
- Evaluation of model performance
- Saving and loading the trained model
- Visualization of training results

## Techniques Used

### Data Preprocessing

- Normalization of image pixel values to the range [0, 1]
- Conversion of labels to one-hot encoding

### Model Definition

- Use of `Sequential` model from Keras
- Multiple convolutional layers with ReLU activation
- MaxPooling layers to reduce spatial dimensions
- Dropout layers to prevent overfitting
- Flattening layer to convert 2D feature maps to 1D feature vectors
- Dense layers for classification

### Training

- Compilation of the model with RMSprop optimizer and categorical cross-entropy loss
- Training the model with a validation split to monitor performance
- Saving the trained model to disk

### Evaluation

- Plotting training and validation loss and accuracy
- Loading the saved model for inference

## Installation

To run this project, you need to have Python and the following libraries installed:

- TensorFlow
- Keras
- NumPy
- Matplotlib

You can install the required libraries using pip:

```bash
pip install tensorflow keras numpy matplotlib