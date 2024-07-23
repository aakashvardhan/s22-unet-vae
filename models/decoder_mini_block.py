import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from .double_conv2D import DoubleConv2D


class DecoderMiniBlock(LightningModule):
    def __init__(
        self, in_channels, out_channels, channel_expansion="upsample", dropout=0
    ):
        super().__init__()
        self.dropout = nn.Dropout2d(dropout)
        self.double_conv2d = DoubleConv2D(in_channels + out_channels, out_channels)

        # Set up the upsampling method
        if channel_expansion == "upsample":
            self.ce = nn.Upsample(
                scale_factor=None, mode="bilinear", align_corners=True
            )
        elif channel_expansion == "transposed_conv":
            self.ce = nn.ConvTranspose2d(
                in_channels, out_channels // 2, kernel_size=2, stride=2
            )
        else:
            raise Exception("Invalid Channel Expansion Method!")

    def forward(self, x, skip):
        # Dynamically adjust the size of x to match that of skip
        if hasattr(self, "ce") and isinstance(self.ce, nn.Upsample):
            self.ce.size = (skip.size(2), skip.size(3))
            x = self.ce(x)
        else:
            x = self.ce(x)

        # print(f"x shape: {x.shape}, skip shape: {skip.shape}")

        # Concatenate along the channel dimension
        x = torch.cat([x, skip], dim=1)
        x = self.double_conv2d(x)
        x = self.dropout(x)
        return x
