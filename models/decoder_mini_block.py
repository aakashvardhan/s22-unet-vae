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
        self.double_conv2d = DoubleConv2D(in_channels + out_channels, out_channels)
        self.dropout = nn.Dropout2d(dropout)

        # Applying Channel Expansion Method
        if channel_expansion == "upsample":
            self.ce = nn.Upsample(
                scale_factor=None, mode="bilinear", align_corners=True
            )
        elif channel_expansion == "transposed_conv":
            self.ce = nn.ConvTranspose2d(
                out_channels, out_channels // 2, kernel_size=2, stride=2
            )
        else:
            raise ValueError("Invalid Channel Expansion Method!")

    def forward(self, x, skip):
        if hasattr(self, "ce") and isinstance(self.ce, nn.Upsample):
            self.ce.size = (skip.size(2), skip.size(3))
            x = self.ce(x)
        else:
            x = self.ce(x)

        # Concatenate along the channel dimension
        x = torch.cat([x, skip], dim=1)
        x = self.double_conv2d(x)
        x = self.dropout(x)

        return x
