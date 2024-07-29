# Building a U-NET model for image segmentation

## Introduction
U-Net is a convolutional neural network architecture that is used for image segmentation. The network is based on the fully convolutional network and its architecture was inspired by the U-Net paper. The U-Net architecture is built upon the FCN and modified in such a way that it yields better segmentation in medical imaging. The architecture consists of a contracting path to capture context and a symmetric expanding path that enables precise localization. The network is trained end-to-end using Oxford IIIT Pet Dataset. The dataset consists of images of cats and dogs and the goal is to segment the images into two classes: cat and dog.

## Training Strategies

The U-Net model is trained using the following strategies:

- `Max Pooling(MP)`: The model uses max pooling layers to downsample the input image.
- `Transpose Convolution(Tr)`: The model uses transpose convolution layers to upsample the input image.
- `Binary Cross Entropy(BCE)`: The model uses binary cross entropy loss function to calculate the loss for binary segmentation.
- `Dice Loss`: The model uses a custom multiclass dice loss function.
- `Strided Convolution(StrConv)`: The model uses strided convolution layers to downsample the input image by a factor of 2.
- `Upsampling(Ups)`: The model uses upsampling layers to upsample the input image by a factor of 2.

The model is trained using the following configurations:

- MP+Tr+BCE
- MP+Tr+Dice Loss
- StrConv+Tr+BCE
- StrConv+Ups+Dice Loss

## Metrics & Visualizations

[View my W&B Report](https://api.wandb.ai/links/akv1000/53nprd7u)

