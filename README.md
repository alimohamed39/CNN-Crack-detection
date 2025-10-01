# CNN-Crack-detection
CNN to detect cracks in images

This project implements a binary image classification model to detect wall cracks.
Images are classified as either:

Positive (Cracked)

Negative (No Crack)

A pretrained ResNet-18 model from PyTorch was fine-tuned on a dataset of 224×224 images.
The model was trained with the Adam optimizer and a learning rate of 0.01, achieving a test accuracy of ~97%.

📌 Project Description

The goal of this project is to build an automated system that can identify cracks in walls from images, which can help in structural health monitoring and preventive maintenance.

Workflow:

Exploratory Data Analysis (EDA)

Distribution of cracked vs non-cracked images

Sample visualization of wall surfaces

Dataset balancing and preprocessing insights

Data Preparation

All images resized to 224×224

Normalization applied using ImageNet statistics (mean & std)

Model

Pretrained ResNet-18 used as the base

Final fully connected layer modified for binary classification

Optimized with Adam (lr=0.01)

Loss: CrossEntropyLoss

Training & Evaluation

Model trained on labeled dataset of wall cracks

Achieved ~97% accuracy on the test set

Supports per-class accuracy reporting
