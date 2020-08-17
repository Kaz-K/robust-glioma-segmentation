import torch
import torch.nn as nn


class Normalize(nn.Module):
    type = 'groupnorm'

    def __init__(self, num_features):
        super().__init__()

        if self.type == 'none':
            self.norm = lambda x: x

        elif self.type == 'batchnorm':
            self.norm = nn.BatchNorm3d(num_features)

        elif self.type == 'instancenorm':
            self.norm = nn.InstanceNorm3d(num_features)

        elif self.type == 'groupnorm':
            self.norm = nn.GroupNorm(num_groups=8, num_channels=num_features)

    def forward(self, x):
        return self.norm(x)


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            Normalize(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels, out_channels, 3, 1, 1, bias=True),
            Normalize(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, 1, 1, bias=True),
        )

        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, 1, 0, bias=True),
            )
        else:
            self.downsample = lambda x: x

    def forward(self, x):
        return self.downsample(x) + self.conv(x)


class UpBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 1, 1, 0, bias=True),
            nn.Upsample(scale_factor=2, mode='trilinear'),
        )

    def forward(self, x, enc):
        x = self.up(x)
        return x + enc


class ResUNet(nn.Module):

    def __init__(self, input_dim=4, output_dim=3, filters=[32, 64, 128, 256]):
        super().__init__()

        self.init_conv = nn.Conv3d(input_dim, filters[0], 3, 1, 1, bias=True)
        self.dropout = nn.Dropout3d(p=0.2)
        self.enc_block_0 = ConvBlock(filters[0], filters[0])
        self.enc_down_1 = nn.Conv3d(filters[0], filters[1], 3, 2, 1, bias=True)
        self.enc_block_1 = nn.Sequential(
            ConvBlock(filters[1], filters[1]),
            ConvBlock(filters[1], filters[1]),
        )
        self.enc_down_2 = nn.Conv3d(filters[1], filters[2], 3, 2, 1, bias=True)
        self.enc_block_2 = nn.Sequential(
            ConvBlock(filters[2], filters[2]),
            ConvBlock(filters[2], filters[2]),
        )
        self.enc_down_3 = nn.Conv3d(filters[2], filters[3], 3, 2, 1, bias=True)
        self.enc_block_3 = nn.Sequential(
            ConvBlock(filters[3], filters[3]),
            ConvBlock(filters[3], filters[3]),
            ConvBlock(filters[3], filters[3]),
            ConvBlock(filters[3], filters[3]),
        )

        self.dec_up_2 = UpBlock(filters[3], filters[2])
        self.dec_block_2 = ConvBlock(filters[2], filters[2])
        self.dec_up_1 = UpBlock(filters[2], filters[1])
        self.dec_block_1 = ConvBlock(filters[1], filters[1])
        self.dec_up_0 = UpBlock(filters[1], filters[0])
        self.dec_block_0 = ConvBlock(filters[0], filters[0])
        self.dec_end = nn.Conv3d(filters[0], output_dim, 1, 1, 0, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.init_conv(x)
        x = self.dropout(x)
        x = self.enc_block_0(x)
        e_0 = x
        x = self.enc_down_1(x)
        x = self.enc_block_1(x)
        e_1 = x
        x = self.enc_down_2(x)
        x = self.enc_block_2(x)
        e_2 = x
        x = self.enc_down_3(x)
        x = self.enc_block_3(x)
        x = self.dec_up_2(x, e_2)
        x = self.dec_block_2(x)
        x = self.dec_up_1(x, e_1)
        x = self.dec_block_1(x)
        x = self.dec_up_0(x, e_0)
        x = self.dec_block_0(x)
        x = self.dec_end(x)
        x = self.sigmoid(x)
        return x
