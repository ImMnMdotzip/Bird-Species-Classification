# Bird Species Classification Using Spiking Neural Networks

## Overview

This project presents the implementation of a bird species classification system using a Spiking Neural Network (SNN). The model is developed with PyTorch and the snnTorch framework and is designed to classify bird images after preprocessing them to a resolution of 96 × 96 pixels. The project is executed in the Google Colab environment with the dataset stored and accessed through Google Drive.

The primary objective of this project is to investigate the applicability of Spiking Neural Networks to image classification tasks while demonstrating a complete deep learning workflow, including data preprocessing, model training, and performance evaluation.

## Objectives

- Develop a bird species classification model based on Spiking Neural Networks.
- Preprocess image data for efficient model training.
- Train and evaluate the classification model using PyTorch and snnTorch.
- Execute the entire workflow in Google Colab with datasets stored on Google Drive.

## Technologies

- Python
- PyTorch
- snnTorch
- Torchvision
- Google Colab
- Google Drive
- NumPy
- Matplotlib

## Project Structure

```
Bird-Species-Classification/
│
├── notebooks/
│   └── SNN_96x96_Birds.ipynb
│
└── README.md
```

## Dataset

The dataset consists of bird images organized into class-specific directories. Images are resized to **96 × 96 pixels** before training to reduce computational complexity and improve training efficiency. The dataset is stored in Google Drive and accessed directly from Google Colab during model execution.

Example dataset location:

```python
/content/drive/MyDrive/birds/Image_Dataset/
```

## Model Architecture

The classification model is implemented using a Spiking Neural Network consisting of:

- Convolutional layers for feature extraction
- Leaky Integrate-and-Fire (LIF) neurons
- Fully connected classification layers
- Surrogate gradient learning for backpropagation

The model is trained using the Adam optimizer and optimized with a classification loss function provided by the snnTorch framework.

## Methodology

The implementation follows the standard deep learning pipeline:

1. Mount Google Drive.
2. Load the image dataset.
3. Preprocess images by resizing them to 96 × 96 pixels.
4. Split the dataset into training and testing sets.
5. Train the Spiking Neural Network.
6. Evaluate the trained model using the testing dataset.

## Results

The notebook includes:

- Data preprocessing
- Model training
- Loss monitoring
- Model evaluation
- Classification performance on the testing dataset

## Execution Environment

The project was developed and executed using:

- Google Colab
- Google Drive for dataset storage
- GPU acceleration (when available)

## Author

M n M
