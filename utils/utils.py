import cv2
import numpy as np
from PIL.Image import Image

def draw_bbox(img, target):
    # Convert img to numpy if PIL Image
    if isinstance(img, Image):
        img = np.array(img)
    result_img = img.copy()
    boxes, labels = target["boxes"], target["labels"]
    # 20 Colors for 20 classes
    np.random.seed(42)
    box_color = [tuple(np.random.randint(0, 256, size=3).tolist()) for _ in range(20)]
    VOC_CLASSES = [
        "aeroplane", "bicycle", "bird", "boat", "bottle",
        "bus", "car", "cat", "chair", "cow", "diningtable",
        "dog", "horse", "motorbike", "person", "pottedplant",
        "sheep", "sofa", "train", "tvmonitor"
    ]
    for i in range(len(boxes)):
        box, label = boxes[i], labels[i]
        result_img = cv2.rectangle(result_img, box[[0, 1]], box[[2, 3]], box_color[label], 3)
        # 0 for background
        class_name = VOC_CLASSES[label - 1]
        cv2.putText(img=result_img, org=box[[0, 1]], text=class_name, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=box_color[label], thickness=1)
    
    return result_img