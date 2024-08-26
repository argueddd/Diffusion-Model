import torch
from torch import nn as nn


class UNet1D(nn.Module):
    def __init__(self, in_channels, out_channels, init_features=32):
        super(UNet1D, self).__init__()

        features = init_features
        self.encoder1 = self._block(in_channels, features)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder2 = self._block(features, features * 2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.bottleneck = self._block(features * 2, features * 4)

        self.upconv2 = nn.ConvTranspose1d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = self._block((features * 2) * 2, features * 2)
        self.upconv1 = nn.ConvTranspose1d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = self._block(features * 2, features)

        self.conv = nn.Conv1d(in_channels=features, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))

        bottleneck = self.bottleneck(self.pool2(enc2))

        dec2 = self.upconv2(bottleneck)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.conv(dec1)

    def _block(self, in_channels, features):
        return nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=features, kernel_size=3, padding=1),
            nn.BatchNorm1d(features),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=features, out_channels=features, kernel_size=3, padding=1),
            nn.BatchNorm1d(features),
            nn.ReLU(inplace=True),
        )
