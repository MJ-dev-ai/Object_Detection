from torch.utils.data import Dataset
import os
from xml.etree import ElementTree as ET
from PIL import Image



class MyTransform:
    def __init__(self, mean=None, std=None):
        # ImageNet 기준 Normalize parameters
        self.mean = mean if mean is not None else [0.485, 0.456, 0.406]
        self.std = std if std is not None else [0.229, 0.224, 0.225]

    def __call__(self, img, boxes, labels):
        import torch
        from torchvision.transforms import functional as F
        """
        img    : PIL.Image
        boxes  : [[xmin, ymin, xmax, ymax], ...] (list of list)
        labels : [class1, class2, ...] (list)
        """
        # --------- 1) PIL Image -> Tensor + Normalize ---------
        img = F.to_tensor(img)  # [C, H, W], float32 [0,1]
        img = F.normalize(img, mean=self.mean, std=self.std)

        # --------- 2) boxes, labels -> Tensor 변환 ---------
        boxes = torch.as_tensor(boxes, dtype=torch.int32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        return img, boxes, labels

class VOCObjectDetection(Dataset):
    def __init__(self, root, image_set="train"):
        # image_set = "train", "val", "trainval"
        split_file = os.path.join(root, "ImageSets", "Main", f"{image_set}.txt")
        if not os.path.isfile(split_file):
            raise FileNotFoundError(f"Split file not found: {split_file}")
        
        with open(split_file) as f:
            image_ids = [line.strip() for line in f if line.strip()]

        annotation_path = os.path.join(root, "Annotations")
        img_path = os.path.join(root, "JPEGImages")

        self.images = []
        self.annotations = []

        for image_id in image_ids:
            image_file = os.path.join(img_path, f"{image_id}.jpg")
            xml_file = os.path.join(annotation_path, f"{image_id}.xml")

            if os.path.exists(image_file) and os.path.exists(xml_file):
                self.images.append(image_file)
                self.annotations.append(xml_file)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")
        annotation_path = self.annotations[idx]
        # PASCAL VOC class names
        VOC_CLASSES = [
            "aeroplane", "bicycle", "bird", "boat", "bottle",
            "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant",
            "sheep", "sofa", "train", "tvmonitor"
        ]

        # Read annotation to find bounding boxes and labels
        xml_tree = ET.parse(annotation_path)
        root = xml_tree.getroot()
        boxes = []
        labels = []
        # Iterate over all object elements in the XML
        for obj in root.findall("object"):
            # Save Labels
            label = obj.find("name").text
            label = VOC_CLASSES.index(obj.find("name").text) + 1 # 0 for background
            labels.append(label)

            # Save Bounding Boxes
            bbox = obj.find("bndbox")
            box = [
                int(bbox.find("xmin").text),
                int(bbox.find("ymin").text),
                int(bbox.find("xmax").text),
                int(bbox.find("ymax").text)
            ]
            boxes.append(box)

        img, boxes, labels = MyTransform()(img, boxes, labels)
        
        return img, {"boxes": boxes, "labels": labels}
    
if __name__ == "__main__":
    dataset = VOCObjectDetection(root="data/VOC2012", image_set="val")
    print(f"Number of samples: {len(dataset)}")
    img, target = dataset[0]
    print(f"Image shape: {img.shape}")
    print(f"Boxes: {target['boxes']}")
    print(f"Labels: {target['labels']}")