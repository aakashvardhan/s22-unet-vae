import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from .double_conv2D import DoubleConv2D


class EncoderMiniBlock(LightningModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        channel_reduction="max_pool",
        dropout=0,
    ):
        super().__init__()
        self.double_conv2d = DoubleConv2D(in_channels, out_channels)
        self.dropout = nn.Dropout2d(dropout)

        if stride > 1:
            # Applying Channel Reduction Method
            if channel_reduction == "max_pool":
                self.cr = nn.MaxPool2d(kernel_size=2, stride=stride)
            elif channel_reduction == "strided_convolution":
                self.cr = nn.Conv2d(
                    out_channels, out_channels, kernel_size=2, stride=stride
                )
            else:
                raise ValueError("Invalid Channel Reduction Method!")

    def forward(self, x):
        x = self.double_conv2d(x)
        x = self.dropout(x)
        skip = x
        if hasattr(self, "cr"):
            x = self.cr(x)
        return x, skip
