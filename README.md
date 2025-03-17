# Accelerating ResNet-18: Transfer Learning and Core ML Integration on Apple M1 Pro GPU

## Overview

This project demonstrates the process of performing transfer learning using ResNet-18, converting the trained model to Core ML format, and leveraging the Apple M1 Pro GPU for accelerated inference.

## Table of Contents

- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [1. Transfer Learning with ResNet-18](#1-transfer-learning-with-resnet-18)
- [2. Converting the Model to Core ML Format](#2-converting-the-model-to-core-ml-format)
- [3. Accelerating Inference on Apple M1 Pro GPU](#3-accelerating-inference-on-apple-m1-pro-gpu)
- [References](#references)

## Introduction

Transfer learning enables us to adapt pre-trained models to new tasks with limited data, reducing training time and computational resources. In this project, we fine-tune ResNet-18 on a custom dataset, convert the model to Core ML format, and utilize the Apple M1 Pro GPU for efficient on-device inference.

## Prerequisites

- Python 3.x
- PyTorch
- torchvision
- coremltools
- An Apple device with M1 Pro chip

## 1. Transfer Learning with ResNet-18

We begin by fine-tuning ResNet-18 on our target dataset. This involves:

- Loading a pre-trained ResNet-18 model.
- Freezing the existing layers to retain learned features.
- Modifying the final fully connected layer to match the number of classes in the new dataset.
- Defining data transformations and loading the dataset.
- Setting up the loss function and optimizer.
- Training the modified model on the new dataset.

For a detailed tutorial on transfer learning with ResNet-18, refer to the [PyTorch Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html).

## 2. Converting the Model to Core ML Format

After training, we convert the model to Core ML format using `coremltools`. The steps include:

- Setting the model to evaluation mode.
- Tracing the model with a dummy input to capture the model graph.
- Using `coremltools` to convert the traced model to Core ML format.
- Saving the Core ML model for deployment.

For more information on converting PyTorch models to Core ML, see the [Apple Core ML Tools Guide](https://apple.github.io/coremltools/docs-guides/source/convert-pytorch.html).

## 3. Accelerating Inference on Apple M1 Pro GPU

To leverage the Apple M1 Pro GPU for inference, we use the Core ML model in an iOS or macOS application. Core ML automatically utilizes the available hardware, including the GPU, for efficient inference. The process involves:

- Loading the Core ML model within the application.
- Preparing the input data in the required format.
- Performing inference using the model.
- Processing the output as needed for the application.

For guidance on deploying Core ML models, consult the [Core ML Documentation](https://developer.apple.com/machine-learning/core-ml/).

## References

- [Transfer Learning for Computer Vision Tutorial - PyTorch](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [Converting from PyTorch — Guide to Core ML Tools - Apple](https://apple.github.io/coremltools/docs-guides/source/convert-pytorch.html)
- [Core ML - Machine Learning - Apple Developer](https://developer.apple.com/machine-learning/core-ml/)
- [Training ResNet18 from Scratch using PyTorch](https://debuggercafe.com/training-resnet18-from-scratch-using-pytorch/)

---

This README provides a comprehensive guide to performing transfer learning with ResNet-18, converting the model to Core ML format, and accelerating inference on the Apple M1 Pro GPU. For detailed explanations and additional resources, please refer to the provided references.

