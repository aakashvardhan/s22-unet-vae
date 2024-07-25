import torch.nn as nn
from lightning import LightningModule
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.encoder_mini_block import EncoderMiniBlock
from models.decoder_mini_block import DecoderMiniBlock


class UNet(LightningModule):

    def __init__(self, config):
        super().__init__()

        # Contraction / Encoding Block
        channel_reduction = config.compression_method
        in_channels = config.in_channels
        n_filters = config.num_filters
        dropout = config.dropout_rate
        out_channels = config.out_channels

        self.enc1 = EncoderMiniBlock(
            in_channels,
            n_filters // 8,
            dropout=dropout,
            channel_reduction=channel_reduction,
        )

        self.enc2 = EncoderMiniBlock(
            n_filters // 8,
            n_filters // 4,
            dropout=dropout,
            channel_reduction=channel_reduction,
        )

        self.enc3 = EncoderMiniBlock(
            n_filters // 4,
            n_filters // 2,
            dropout=dropout,
            channel_reduction=channel_reduction,
        )

        self.enc4 = EncoderMiniBlock(
            n_filters // 2,
            n_filters,
            dropout=dropout,
            channel_reduction=channel_reduction,
        )

        # Expansion / Decoding Block
        channel_expansion = config.expansion_method
        self.dec1 = DecoderMiniBlock(
            n_filters,
            n_filters // 2,
            dropout=dropout,
            channel_expansion=channel_expansion,
        )

        self.dec2 = DecoderMiniBlock(
            n_filters // 2,
            n_filters // 4,
            dropout=dropout,
            channel_expansion=channel_expansion,
        )

        self.dec3 = DecoderMiniBlock(
            n_filters // 4,
            n_filters // 8,
            dropout=dropout,
            channel_expansion=channel_expansion,
        )

        # Final Layer
        self.final_layer = nn.Conv2d(n_filters // 8, out_channels, kernel_size=1)

        # Assert in_channels is 3 and in_channels == out_channels
        assert in_channels == 3, "in_channels must be 3"
        assert in_channels == out_channels, "in_channels must be equal to out_channels"

    def forward(self, x):
        # Contraction / Encoding Block
        x, skip1 = self.enc1(x)
        x, skip2 = self.enc2(x)
        x, skip3 = self.enc3(x)
        x, _ = self.enc4(x)

        # Expansion / Decoding Block
        x = self.dec1(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec3(x, skip1)
        x = self.final_layer(x)

        return x


# if __name__ == "__main__":
#     from config import UNetConfig, load_config, update_config
#     from torchsummary import summary

#     config = UNetConfig()

#     json_data = load_config("training_1.json")
#     config = update_config(config, json_data)

#     net = UNet(config)

#     summary(net, (3, 240, 240))
