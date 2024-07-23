from lightning import LightningModule
import torch.nn as nn


class DoubleConv2D(LightningModule):
    """
    DoubleConv2D module consists of two consecutive convolutional layers followed by ReLU activation and batch normalization.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int, optional): Size of the convolutional kernel. Defaults to 3.
        stride (int, optional): Stride value for the convolutional layers. Defaults to 1.
        padding (int, optional): Padding value for the convolutional layers. Defaults to 1.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        """
        Forward pass of the DoubleConv2D module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the convolutional layers.
        """
        return self.conv(x)
