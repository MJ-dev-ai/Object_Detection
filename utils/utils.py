import cv2
import torch
import numpy as np
from PIL import Image

def draw_bbox(img, target):
    # Convert img to numpy if PIL Image or tensor
    if isinstance(img, Image.Image):
        img = np.array(img)
    elif isinstance(img, torch.tensor):
        img = (img * 255.0).permute(1,2,0).numpy().astype(np.uint8)
    result_img = img.copy()
    boxes, labels = target["boxes"], target["labels"]
    # 20 Colors for 20 classes
    box_color = [tuple(np.random.randint(0, 256, size=3).tolist()) for _ in range(20)]
    VOC_CLASSES = [
        "aeroplane", "bicycle", "bird", "boat", "bottle",
        "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant",
        "sheep", "sofa", "train", "tvmonitor"
    ]
    for i in range(len(boxes)):
        box, label = boxes[i], labels[i]
        pt1 = (int(box[0]), int(box[1]))
        pt2 = (int(box[2]), int(box[3]))
        print("Label:", label, "Color:", box_color[int(label) - 1], "pt1:", pt1, "pt2:", pt2)
        result_img = cv2.rectangle(img=result_img, pt1=box[:2], pt2=box[2:], color=box_color[label - 1], thickness=3)
        # 0 for background
        class_name = VOC_CLASSES[label - 1]
        cv2.putText(img=result_img, org=box[:2], text=class_name, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=box_color[label - 1], thickness=1)
    
    return result_img