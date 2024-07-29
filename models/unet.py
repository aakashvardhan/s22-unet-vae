import torch.nn as nn
import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DoubleConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class EncoderMiniBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, dropout=0.0, channel_reduction="max_pool"
    ):
        super().__init__()

        self.conv = DoubleConv2D(in_channels, out_channels, dropout=dropout)

        if channel_reduction == "max_pool":
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        elif channel_reduction == "conv_stride":
            self.pool = nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=2, padding=1
            )
        else:
            self.pool = None

    def forward(self, x):
        x = self.conv(x)
        skip = x

        if self.pool is not None:
            x = self.pool(x)

        return x, skip


class DecoderMiniBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, dropout=0.0, channel_expansion="upsample"
    ):
        super().__init__()

        self.conv = DoubleConv2D(in_channels, out_channels, dropout=dropout)

        if channel_expansion == "upsample":
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.up_conv = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
        elif channel_expansion == "conv_transpose":
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=3, stride=2, padding=1, output_padding=1
            )
        else:
            self.up = None

    def forward(self, x, skip):
        x = self.up(x)

        if isinstance(self.up, nn.Upsample):
            x = self.up_conv(x)
            # print(f"x shape after upsample: {x.shape}")

        # print(f"x shape after up: {x.shape}")
        # print(f"skip shape: {skip.shape}")
        x = torch.cat([skip, x], dim=1)
        # print(f"x shape after cat: {x.shape}")
        x = self.conv(x)
        return x


class UNet(nn.Module):

    def __init__(self, config, debug=False):
        super().__init__()

        # Contraction / Encoding Block
        self.debug = debug

        channel_reduction = config.channel_reduction_method
        channel_expansion = config.channel_expansion_method
        in_channels = config.in_channels
        n_filters = config.num_filters
        dropout = config.dropout_rate
        out_channels = config.out_channels

        self.enc1 = EncoderMiniBlock(
            in_channels,
            n_filters,
            dropout=dropout,
            channel_reduction=channel_reduction,
        )

        self.enc2 = EncoderMiniBlock(
            n_filters,
            n_filters * 2,
            dropout=dropout,
            channel_reduction=channel_reduction,
        )

        self.enc3 = EncoderMiniBlock(
            n_filters * 2,
            n_filters * 4,
            dropout=0.1,
            channel_reduction=channel_reduction,
        )

        self.enc4 = EncoderMiniBlock(
            n_filters * 4,
            n_filters * 8,
            dropout=0.1,
            channel_reduction=None,
        )

        # Expansion / Decoding Block
        self.dec1 = DecoderMiniBlock(
            n_filters * 8,
            n_filters * 4,
            dropout=dropout,
            channel_expansion=channel_expansion,
        )

        self.dec2 = DecoderMiniBlock(
            n_filters * 4,
            n_filters * 2,
            dropout=0.1,
            channel_expansion=channel_expansion,
        )

        self.dec3 = DecoderMiniBlock(
            n_filters * 2,
            n_filters,
            dropout=0.1,
            channel_expansion=channel_expansion,
        )

        # Final Layer
        self.final_layer = nn.Conv2d(n_filters, out_channels, kernel_size=1)

        # Assert in_channels is 3 and in_channels == out_channels
        assert in_channels == 3, "in_channels must be 3"
        assert in_channels == out_channels, "in_channels must be equal to out_channels"

    def forward(self, x):
        # Contraction / Encoding Block
        if self.debug:
            print(f"Input shape: {x.shape}")
        x, skip1 = self.enc1(x)
        if self.debug:
            print(f"enc1 shape: {x.shape}")
            print(f"skip1 shape: {skip1.shape}")
        x, skip2 = self.enc2(x)
        if self.debug:
            print(f"enc2 shape: {x.shape}")
            print(f"skip2 shape: {skip2.shape}")
        x, skip3 = self.enc3(x)
        if self.debug:
            print(f"enc3 shape: {x.shape}")
            print(f"skip3 shape: {skip3.shape}")
        x, _ = self.enc4(x)
        if self.debug:
            print(f"enc4 shape: {x.shape}")

        # Expansion / Decoding Block
        x = self.dec1(x, skip3)
        if self.debug:
            print(f"dec1 shape: {x.shape}")
        x = self.dec2(x, skip2)
        if self.debug:
            print(f"dec2 shape: {x.shape}")
        x = self.dec3(x, skip1)
        if self.debug:
            print(f"dec3 shape: {x.shape}")
        x = self.final_layer(x)
        if self.debug:
            print(f"final_layer shape: {x.shape}")

        return x


if __name__ == "__main__":
    from config import UNetConfig, load_config, update_config
    from torchsummary import summary

    config = UNetConfig()

    json_data = load_config("training_4.json")
    config = update_config(config, json_data)
    print(config)

    net = UNet(config, debug=True)
    # print(net)
    summary(net, (3, 240, 240))
