import torch
from torch import nn
# Conv block with SiLU activation
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, groups=1, act=True):
        super().__init__()
        if padding is None: padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
            stride=stride, padding=padding, groups=groups, bias=False
            )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if act else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    
# Bottleneck Block for C3
class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True):
        super().__init__()
        # Channel reduction with 1x1 Conv Layer
        self.conv1 = Conv(c1, c2)
        self.conv2 = Conv(c2, c2, 3)
        # Skip Connection Check
        self.skip = shortcut

    def forward(self, x):
        x = self.conv1(x)
        x2 = self.conv2(x)
        return x + x2 if self.skip else x2

# C3 block with n bottleneck layer
class C3Block(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True):
        super().__init__()
        # Hidden Layer Channel = C1 / 2
        c_hidden = c1 // 2
        self.conv1 = Conv(c1, c_hidden)
        self.conv2 = Conv(c1, c_hidden)
        self.conv3 = Conv(c_hidden * 2, c2)
        # Repeat Bottleneck Block n times
        self.bottleneck = nn.Sequential(*[Bottleneck(c_hidden, c_hidden, shortcut) for _ in range(n)])
    
    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bottleneck(x1)
        x2 = self.conv2(x)

        # Concatenate: C1 / 2 + C1 / 2 = C1 channels
        x3 = torch.cat((x1, x2), dim=1)
        x3 = self.conv3(x3)
        return x3

# Concat outputs of conv and maxpool layers with stride 1
class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_hidden = c1 // 2
        self.conv1 = Conv(c1, c_hidden)
        self.maxpool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.conv2 = Conv(c_hidden * 4, c2)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.maxpool(x)
        y2 = self.maxpool(y1)
        y3 = self.maxpool(y2)

        return self.conv2(torch.cat([x, y1, y2, y3], dim=1))