# task2
# README - CNN for CIFAR-10 Classification

## Project Overview
This project implements a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The goal is to train a model that can accurately classify these images.

## Features
- **CNN Architecture**: The model includes two convolutional layers, max-pooling, dropout for regularization, and two fully connected layers.
- **Data Augmentation**: Random horizontal flips and rotations are applied to the training data to improve generalization.
- **Training and Evaluation**: The model is trained using the Adam optimizer and evaluated on a separate test set.

## Requirements
- Python 3.x
- PyTorch
- torchvision
- matplotlib

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```
2. Install the required packages:
   ```bash
   pip install torch torchvision matplotlib
   ```

## Usage
1. Run the script to train and evaluate the model:
   ```bash
   python task2.py
   ```
2. The script will:
   - Download the CIFAR-10 dataset.
   - Train the CNN model for 15 epochs.
   - Evaluate the model on the test set and print the accuracy.
   - Plot the training loss over epochs.

## Code Structure
- **CNN Class**: Defines the architecture of the convolutional neural network.
- **Data Loading**: Loads and preprocesses the CIFAR-10 dataset with transformations.
- **Training Loop**: Trains the model using the training data and records the loss.
- **Evaluation**: Evaluates the model's performance on the test data and calculates accuracy.
- **Visualization**: Plots the training loss over epochs using matplotlib.

## Results
After training for 15 epochs, the model achieves a test accuracy of approximately **XX.XX%** (accuracy will vary based on training). The training loss plot shows the decrease in loss over epochs, indicating successful learning.

## Future Improvements
- Experiment with deeper architectures (e.g., adding more convolutional layers).
- Try different hyperparameters (learning rate, batch size, etc.).
- Implement advanced techniques like batch normalization or residual connections.
