import torch
import torch.nn as nn
import torch.nn.functional as F

# -------- CBAM Core Modules --------
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x))

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention()

    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out

# -------- Wrapper for Bottleneck Blocks --------
class CBAMWrapper(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block
        # Choose conv3 for Bottleneck (ResNet50/101)
        self.cbam = CBAM(in_planes=block.conv3.out_channels)

    def forward(self, x):
        out = self.block(x)
        out = self.cbam(out)
        return out

# -------- Injection Function --------
def inject_cbam_into_layer(target, device):
    """
    Inject CBAM into:
    - The whole DeepLabV3 model (target.backbone.layer4),
    - The entire layer4 Sequential,
    - Or into each block inside layer4 (wrap with CBAMWrapper).
    """
    if hasattr(target, "backbone") and hasattr(target.backbone, "layer4"):
        # Case 1: full model
        print("Injecting CBAM into model.backbone.layer4...")
        in_channels = target.backbone.layer4[-1].conv3.out_channels
        cbam = CBAM(in_channels).to(device)
        target.backbone.layer4.add_module("cbam", cbam)

    elif isinstance(target, nn.Sequential):
        # Case 2: layer4 Sequential
        print("Injecting CBAM into each block of layer4 Sequential...")
        for i, block in enumerate(target):
            target[i] = CBAMWrapper(block).to(device)

    elif hasattr(target, "conv3"):
        # Case 3: a single Bottleneck block
        print("Injecting CBAM into a Bottleneck block...")
        return CBAMWrapper(target).to(device)

    else:
        raise ValueError("inject_cbam_into_layer: unsupported input type.")




