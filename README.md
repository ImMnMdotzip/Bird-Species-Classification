# Bird Species Classification Using Spiking Neural Networks

## Overview

This project presents the implementation of an automated bird species classification system using a **Spiking Neural Network (SNN)**. Built with **PyTorch** and **snnTorch**, the model classifies **7 bird species** at an input resolution of **128 × 128 pixels** and achieves an overall validation accuracy of **89.5%** after 30 training epochs.

Unlike traditional cloud-only workflows, this project runs locally in **Visual Studio Code (VS Code)** with GPU acceleration (CUDA 12.8) while programmatically fetching and synchronizing datasets via the **Google Drive API**.

## Objectives

- Develop and train a deep Spiking Neural Network (SNN) for multi-class bird species classification.
- Automate dataset retrieval and management using the Google Drive API in a local Python environment.
- Leverage GPU acceleration (NVIDIA Blackwell / CUDA 12.8) for efficient surrogate gradient backpropagation.
- Export trained model weights (snn_birds_best.pt) and provide an inference pipeline (predict()) for classifying unseen bird images.

## Technologies

- **Language & Environment:** Python 3.12, Visual Studio Code, Jupyter Notebook, uv Package Manager
- **Deep Learning:** PyTorch (cu128), snnTorch, Torchvision
- **Data & Visualization:** NumPy, Pillow (PIL), Matplotlib, scikit-learn
- **Cloud Integration:** Google Drive API (google-api-python-client, google-auth-oauthlib)
- **Hardware Acceleration:** NVIDIA GeForce RTX 5060 Ti (CUDA 12.8 / sm_120)

## Project Structure

- .vscode/settings.json - VS Code workspace settings
- outputs/snn_birds_best.pt - Best trained SNN model weights (89.5% accuracy)
- test_images/test_kingfisher.jpg - Sample test image (Common-Kingfisher)
- test_images/test_peacock.png - Sample test image (Indian-Peacock)
- SNN_Birds_VSCode_GoogleDrive_API.ipynb - Main training, evaluation & inference notebook
- .gitignore - Excludes .venv/, data/Image_Dataset/, credentials.json, and token.json
- README.md - Project documentation

## Dataset

The dataset consists of RGB bird images across **7 distinct species** (240 validation samples per class; 1,680 total validation images), preprocessed and resized to **128 × 128 pixels**:

1. Common-Kingfisher
2. Common-Myna
3. House-Crow
4. Indian-Peacock
5. Indian-Pitta
6. Ruddy-Shelduck
7. Sarus-Crane

During execution, the notebook authenticates via OAuth 2.0 (credentials.json) and automatically downloads the dataset from Google Drive into data/Image_Dataset/.

## Model Architecture & Methodology

- **Feature Extraction:** Convolutional layers (Conv2d, Batch Normalization, Pooling) to extract spatial representations.
- **Spiking Dynamics:** Leaky Integrate-and-Fire (**LIF**) neurons operating across discrete time steps.
- **Training Strategy:** Surrogate gradient descent with the **Adam** optimizer and Cross-Entropy loss over **30 epochs**.

### Workflow Pipeline

1. Authenticate with the Google Drive API and download/verify data/Image_Dataset/.
2. Apply image resizing (128 × 128), tensor conversion, and normalization.
3. Split data into training and validation sets.
4. Train the SNN model on GPU (cuda) and checkpoint the best weights to outputs/snn_birds_best.pt.
5. Evaluate per-class Precision, Recall, F1-score, and Confusion Matrix.
6. Run Top-k inference on unseen images inside test_images/ using predict().

## Results

After 30 epochs of training, the SNN achieved **89.5% overall accuracy** across 1,680 validation images:

| Bird Species | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Common-Kingfisher** | 0.892 | 0.933 | 0.912 | 240 |
| **Common-Myna** | 0.918 | 0.842 | 0.878 | 240 |
| **House-Crow** | 0.842 | 0.821 | 0.831 | 240 |
| **Indian-Peacock** | 0.920 | 0.863 | 0.890 | 240 |
| **Indian-Pitta** | 0.904 | 0.938 | 0.920 | 240 |
| **Ruddy-Shelduck** | 0.970 | 0.933 | 0.951 | 240 |
| **Sarus-Crane** | 0.833 | 0.938 | 0.882 | 240 |
| **Overall Accuracy / Macro Avg** | **0.897** | **0.895** | **0.895** | **1680** |

## Getting Started

1. **Clone the repository:** git clone https://github.com/ImMnMdotzip/Bird-Species-Classification.git
2. **Set up environment (using uv):**
   - uv venv --python 3.12
   - uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
   - uv pip install snntorch matplotlib scikit-learn pillow google-api-python-client google-auth-httplib2 google-auth-oauthlib ipykernel
3. **Run Inference or Retrain:**
   - **Quick Inference:** Open SNN_Birds_VSCode_GoogleDrive_API.ipynb, load pre-trained weights from outputs/snn_birds_best.pt, and run predict() on images inside test_images/.
   - **Full Training:** Place your Google Cloud OAuth credentials.json in the project root and execute all cells from top to bottom.

## Author

**ImMnMdotzip**
