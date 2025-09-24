import torch
import torch.nn as nn
import torch.nn.functional as F

# ===== Double Conv Block: two 3x3 convs + ReLU =====
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

# ===== U-Net main structure =====
class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=1):
        """
        Args:
            in_channels: input channels (e.g., 3 for RGB, 4 for NIR+RGB)
            out_channels: output channels (usually 1 for binary mask)
        """
        super().__init__()

        # Encoder (downsampling)
        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.down4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck
        self.bottom = DoubleConv(512, 1024)

        # Decoder (upsampling + skip connections)
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64)

        # Final 1x1 conv
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        c1 = self.down1(x)
        p1 = self.pool1(c1)
        c2 = self.down2(p1)
        p2 = self.pool2(c2)
        c3 = self.down3(p2)
        p3 = self.pool3(c3)
        c4 = self.down4(p3)
        p4 = self.pool4(c4)

        # Bottleneck
        c5 = self.bottom(p4)

        # Decoder with skip connections
        u1 = self.up1(c5)
        u1 = self.conv1(torch.cat([u1, c4], dim=1))
        u2 = self.up2(u1)
        u2 = self.conv2(torch.cat([u2, c3], dim=1))
        u3 = self.up3(u2)
        u3 = self.conv3(torch.cat([u3, c2], dim=1))
        u4 = self.up4(u3)
        u4 = self.conv4(torch.cat([u4, c1], dim=1))

        out = self.out_conv(u4)
        return out
