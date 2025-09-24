import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block for channel attention.
    It adaptively recalibrates channel-wise feature responses.
    """
    def __init__(self, in_ch, reduction=8):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_ch, in_ch // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(in_ch // reduction, in_ch, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)         # Global average pooling
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b, c, 1, 1)
        return x * y                        # Scale input by learned channel weights
    
class DoubleConv(nn.Module):
    """
    Double convolution block with GroupNorm, LeakyReLU, Dropout, and SEBlock.
    Used in both encoder and decoder paths.
    """
    def __init__(self, in_channels, out_channels, p=0.2):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.LeakyReLU(inplace=True),
            SEBlock(out_channels)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNetPP(nn.Module):
    """
    Simplified UNet++ structure with nested skip pathways.
    Supports flexible input/output channels and uses enhanced DoubleConv blocks.
    """
    def __init__(self, in_channels=4, out_channels=1):
        super(UNetPP, self).__init__()

        # Encoder path
        self.conv00 = DoubleConv(in_channels, 64)
        self.pool0 = nn.MaxPool2d(2)

        self.conv10 = DoubleConv(64, 128)
        self.pool1 = nn.MaxPool2d(2)

        self.conv20 = DoubleConv(128, 256)
        self.pool2 = nn.MaxPool2d(2)

        self.conv30 = DoubleConv(256, 512)
        self.pool3 = nn.MaxPool2d(2)

        self.conv40 = DoubleConv(512, 1024)

        # Nested decoder blocks (partial UNet++ structure)
        self.up31 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv31 = DoubleConv(512 + 512, 512)

        self.up21 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv21 = DoubleConv(256 + 256, 256)

        self.up11 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv11 = DoubleConv(128 + 128, 128)

        self.up01 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv01 = DoubleConv(64 + 64, 64)

        self.up02 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.conv02 = DoubleConv(64 + 64 + 64, 64)  # Concatenates outputs from x01, x00, and upsampled x01

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder path
        x00 = self.conv00(x)
        x10 = self.conv10(self.pool0(x00))
        x20 = self.conv20(self.pool1(x10))
        x30 = self.conv30(self.pool2(x20))
        x40 = self.conv40(self.pool3(x30))

        # Nested decoder path (UNet++ style with deep supervision and dense skip connections)
        x31 = self.conv31(torch.cat([self.up31(x40), x30], dim=1))
        x21 = self.conv21(torch.cat([self.up21(x31), x20], dim=1))
        x11 = self.conv11(torch.cat([self.up11(x21), x10], dim=1))
        x01 = self.conv01(torch.cat([self.up01(x11), x00], dim=1))

        # Second-level skip connection and upsampling
        up_x01 = self.up02(x01)
        up_x01 = F.interpolate(up_x01, size=x00.shape[2:], mode="bilinear", align_corners=True)
        x02 = self.conv02(torch.cat([x01, x00, up_x01], dim=1))

        out = self.final(x02)
        return out
