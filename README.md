# Glomeruli-Classification
This project focuses on the classification of glomeruli using deep learning techniques. The goal is to distinguish between non-globally sclerotic glomeruli and globally sclerotic glomeruli based on image data.
# Deep Learning Models for Binary Classification
This repository contains the implementation of various deep learning models for binary classification tasks using PyTorch. The models used include AlexNet, VGG19, ResNet50, and GoogLeNet. This project aims to evaluate the performance of these models on a binary classification problem and compare their metrics.

# Table of Contents
 1. Introduction
 2. Models Implemented
 3. Results
 4. Requirements
 5. Training and Evaluation
 6. Conclusion
 7. References

# Introduction
Deep learning has shown significant improvements in various computer vision tasks. This repository focuses on implementing different backbone architectures for binary classification tasks and evaluating their performance based on metrics such as accuracy, precision, recall, and F1 score.

# Models Implemented
1. AlexNet
  Test Accuracy: 94.79%
  Test Precision: 80.95%
  Test Recall: 91.28%
  Test F1 Score: 85.80%
2. VGG19
  Training was attempted but the model took too long to train fully within the given constraints.
3. ResNet50
  Training was attempted but the model took too long to train fully within the given constraints.
4. GoogLeNet
  Test Accuracy: 97.92%
  Test Precision: 96.45%
  Test Recall: 91.28%
  Test F1 Score: 93.79%
# Results
  The best performance was achieved using GoogLeNet with a Test F1 Score of 94.69%. Various optimizers and learning rate schedulers were experimented with to achieve these results. The optimizers used include Adam, AdamW, SGD, and RMSProp, with    RMSProp providing the best results. For learning rate scheduling, ReduceOnPlateau, OneCycleLR, and LinearStepLR were tested, with OneCycleLR yielding the best performance.
