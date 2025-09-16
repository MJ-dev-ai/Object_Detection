import torch
from torch import nn
from .common import Conv, C3Block, Focus, SPPF

# Darknet Backbone for YOLOv5n
class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.focus = Focus(3, 12)
        self.conv1 = Conv(12, 16, kernel_size=3, stride=2)
        self.c3_1 = C3Block(16, 16, n=1)

        self.conv2 = Conv(16, 32, kernel_size=3, stride=2)
        self.c3_2 = C3Block(32, 32, n=1)

        self.conv3 = Conv(32, 64, kernel_size=3, stride=2)
        self.c3_3 = C3Block(64, 64, n=2)

        self.conv4 = Conv(64, 128, kernel_size=3, stride=2)
        self.c3_4 = C3Block(128, 128, n=3)

        self.sppf = SPPF(128, 128)

    def forward(self, x):
        x = self.focus(x)
        x = self.conv1(x)
        x = self.c3_1(x)

        x = self.conv2(x)
        y1 = self.c3_2(x)

        x = self.conv3(y1)
        y2 = self.c3_3(x)

        x = self.conv4(y2)
        x = self.c3_4(x)

        y3 = self.sppf(x)

        return (y1, y2, y3)

# Neck
class Neck(nn.Module):
    def __init__(self):
        super().__init__()
        # Top-Down Path
        self.conv1 = Conv(128, 64)
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.c3_1 = C3Block(128, 64, shortcut=False)

        self.conv2 = Conv(64, 32)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.c3_2 = C3Block(64, 32)

        # Botton-Up Path
        self.conv3 = Conv(32, 32, kernel_size=3, stride=2)
        self.c3_3 = C3Block(96, 64, shortcut=False)

        self.conv4 = Conv(64, 64, kernel_size=3, stride=2)
        self.c3_4 = C3Block(128, 128, shortcut=False)

    def forward(self, p3, p4, p5):
        # p3: (32,80,80), p4: (64,40,40), p5: (128,20,20)
        # Top-Down
        y1 = self.up1(self.conv1(p5)) # (64,40,40)
        y1 = torch.cat([y1, p4], dim=1) # (64+64,40,40)
        y1 = self.c3_1(y1) # (64,40,40)

        y2 = self.up2(self.conv2(y1)) # (32,80,80)
        y2 = torch.cat([y2, p3], dim=1) # (32+32,80,80)
        out_1 = self.c3_2(y2) # (32,80,80)

        # Bottom-Up
        y3 = self.conv3(out_1) # (32,40,40)
        y3 = torch.cat([y3, y1], dim=1) # (32+64,40,40)
        out_2 = self.c3_3(y3) # (64,40,40)

        y4 = self.conv4(out_2) # (64,20,20)
        y4 = torch.cat([y4, self.conv1(p5)], dim=1) # (64+64,20,20)
        out_3 = self.c3_4(y4) # (128,20,20)

        return out_1, out_2, out_3  

# Head for Object Detection and Classification
class Head(nn.Module):
    def __init__(self, num_classes=20, anchor=3):
        super().__init__()
        # num_outputs = anchors_per_scale * ((x, y, w, h, objectiveness) + num_classes)
        num_outputs = anchor * (5 + num_classes)
        # 1X1 kernel conv layer for detection
        self.detect_small = nn.Conv2d(32, num_outputs, kernel_size=1)
        self.detect_medium = nn.Conv2d(64, num_outputs, kernel_size=1)
        self.detect_large = nn.Conv2d(128, num_outputs, kernel_size=1)
    
    def forward(self, out_1, out_2, out_3):
        out_small = self.detect_small(out_1) # (75,80,80)
        out_medium = self.detect_medium(out_2) # (75,40,40)
        out_large = self.detect_large(out_3) # (75,20,20)
        return (out_small, out_medium, out_large)

class YOLOv5n(nn.Module):
    def __init__(self, num_classes=20, anchor=3):
        super().__init__()
        self.backbone = Backbone()
        self.neck = Neck()
        self.head = Head(num_classes=num_classes, anchor=anchor)
    
    def forward(self, x):
        p3, p4, p5 = self.backbone(x)
        out_1, out_2, out_3 = self.neck(p3, p4, p5)
        return self.head(out_1, out_2, out_3)

if __name__ == "__main__":
    from torchinfo import summary
    model = YOLOv5n()
    summary(model, input_size=(1, 3, 320, 320))
