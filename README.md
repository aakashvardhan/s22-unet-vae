# Building a U-NET model for image segmentation

## Introduction
U-Net is a convolutional neural network architecture that is used for image segmentation. The network is based on the fully convolutional network and its architecture was inspired by the U-Net paper. The U-Net architecture is built upon the FCN and modified in such a way that it yields better segmentation in medical imaging. The architecture consists of a contracting path to capture context and a symmetric expanding path that enables precise localization. The network is trained end-to-end using Oxford IIIT Pet Dataset. The dataset consists of images of cats and dogs and the goal is to segment the images into two classes: cat and dog.

## Training Strategies

The U-Net model is trained using the following strategies:




## Metrics & Visualizations

<iframe src="https://wandb.ai/akv1000/s22-unet-vae/reports/U-Net-from-Scratch---Vmlldzo4ODUyMjI0" style="border:none;height:1024px;width:100%">