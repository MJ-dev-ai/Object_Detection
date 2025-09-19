import torch
from torch import nn
from .common import Conv, C3Block, SPPF

# Darknet Backbone for YOLOv5n
class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv(3, 16, kernel_size=6, stride=2, padding=2)
        self.conv2 = Conv(16, 32, kernel_size=3, stride=2)
        self.c3_1 = C3Block(32, 32, n=1)

        self.conv3 = Conv(32, 64, kernel_size=3, stride=2)
        self.c3_2 = C3Block(64, 64, n=2)

        self.conv4 = Conv(64, 128, kernel_size=3, stride=2)
        self.c3_3 = C3Block(128, 128, n=2)

        self.conv5 = Conv(128, 256, kernel_size=3, stride=2)
        self.c3_4 = C3Block(256, 256, n=1)

        self.sppf = SPPF(256, 256, k=5)

    def forward(self, x):
        # input size = (3, 320, 320)
        x = self.conv1(x) # (16, 160, 160)
        x = self.conv2(x) # (32, 80, 80)
        x = self.c3_1(x) # (32, 80, 80)

        x = self.conv3(x) # (64, 40, 40)
        y1 = self.c3_2(x) # (64, 40, 40)

        x = self.conv4(y1) # (128, 20, 20)
        y2 = self.c3_3(x) # (128, 20, 20)

        x = self.conv5(y2) # (256, 10, 10)
        x = self.c3_4(x) # (256, 10, 10)

        y3 = self.sppf(x) # (256, 10, 10)

        return (y1, y2, y3)

# Neck
class Head(nn.Module):
    def __init__(self):
        super().__init__()
        # Top-Down Path
        self.conv1 = Conv(256, 128)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.c3_1 = C3Block(256, 128, n=1, shortcut=False)

        self.conv2 = Conv(128, 64)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.c3_2 = C3Block(128, 64, n=1, shortcut=False)

        # Botton-Up Path
        self.conv3 = Conv(64, 64, kernel_size=3, stride=2)
        self.c3_3 = C3Block(192, 128, n=1, shortcut=False)

        self.conv4 = Conv(128, 128, kernel_size=3, stride=2)
        self.c3_4 = C3Block(256, 256, n=1, shortcut=False)

    def forward(self, p3, p4, p5):
        # p3: (64, 40, 40), p4: (128, 20, 20), p5: (256, 10, 10)
        # Top-Down
        y_skip = self.conv1(p5) # (128, 10, 10)
        y1 = self.up1(y_skip) # (128, 10, 10) -> (128, 20, 20)
        y1 = torch.cat([y1, p4], dim=1) # (128+128, 20, 20)
        y1 = self.c3_1(y1) # (128, 20, 20)

        y2 = self.up2(self.conv2(y1)) # (64, 20, 20) -> (64, 40, 40)
        y2 = torch.cat([y2, p3], dim=1) # (64+64, 40, 40)
        out_1 = self.c3_2(y2) # (64, 40, 40) Small

        # Bottom-Up
        y3 = self.conv3(out_1) # (64, 20, 20)
        y3 = torch.cat([y3, y1], dim=1) # (64+128, 20, 20)
        out_2 = self.c3_3(y3) # (128, 20, 20) Medium

        y4 = self.conv4(out_2) # (128, 20, 20)
        y4 = torch.cat([y4, y_skip], dim=1) # (128+128, 10, 10)
        out_3 = self.c3_4(y4) # (256, 10, 10) Large

        return out_1, out_2, out_3  

# Head for Object Detection and Classification
class Detect(nn.Module):
    def __init__(self, num_classes=20, anchor=3):
        super().__init__()
        # num_outputs = anchors_per_scale * ((x, y, w, h, objectiveness) + num_classes)
        num_outputs = anchor * (5 + num_classes)
        # 1X1 kernel conv layer for detection
        self.detect_small = nn.Conv2d(64, num_outputs, kernel_size=1)
        self.detect_medium = nn.Conv2d(128, num_outputs, kernel_size=1)
        self.detect_large = nn.Conv2d(256, num_outputs, kernel_size=1)
    
    def forward(self, out_1, out_2, out_3):
        out_small = self.detect_small(out_1)
        out_medium = self.detect_medium(out_2)
        out_large = self.detect_large(out_3)
        return (out_small, out_medium, out_large)

class YOLOv5n(nn.Module):
    def __init__(self, num_classes=20, anchor=3):
        super().__init__()
        self.backbone = Backbone()
        self.head = Head()
        self.detect = Detect(num_classes=num_classes, anchor=anchor)
    
    def forward(self, x):
        p3, p4, p5 = self.backbone(x)
        out_1, out_2, out_3 = self.head(p3, p4, p5)
        return self.detect(out_1, out_2, out_3)

if __name__ == "__main__":
    from torchinfo import summary
    model = YOLOv5n(num_classes=20, anchor=3)
    dummy_input = torch.randn(1, 3, 320, 320)
    outputs = model(dummy_input)
    print([out.shape for out in outputs])
