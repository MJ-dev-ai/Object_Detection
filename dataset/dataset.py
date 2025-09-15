import torch
from torch.utils.data import Dataset
import os
import cv2
from xml.etree import ElementTree as ET
from PIL import Image
import numpy as np
from .augmentation import MosaicAugmentor

class MyTransform:
    def __init__(self, mean=None, std=None):
        # Normalize parameters from ImageNet
        self.mean = mean if mean is not None else [0.485, 0.456, 0.406]
        self.std = std if std is not None else [0.229, 0.224, 0.225]

    def __call__(self, img):
        from torchvision.transforms import functional as F
        from torchvision.transforms import ColorJitter
        """
        img    : PIL.Image
        boxes  : [[xmin, ymin, xmax, ymax], ...] (list of list)
        labels : [class1, class2, ...] (list)
        """
        # --------- 1) PIL Image -> Tensor + Normalize ---------
        img = F.to_tensor(img)  # [C, H, W], float32 [0,1]
        img = ColorJitter(0.3, 0.3, 0.3)(img)
        img = F.normalize(img, mean=self.mean, std=self.std)

        return img

class VOCObjectDetection(Dataset):
    """
    PASCAL VOC Object Detection Dataset Loader.

    This class loads the PASCAL VOC 2012 dataset and provides it in the format of
    `torch.utils.data.Dataset`.
    
    - Links images with their corresponding XML annotation files and returns them as a `(image, target)` tuple.
    - `target` is a dictionary containing "boxes" and "labels".
    - "boxes" represents the bounding box coordinates of objects,
      while "labels" represents the class indices of those objects.

    Args:
        root (str): Path to the root directory of the VOC dataset.
            Example:
            root/
                Annotations/
                ImageSets/
                JPEGImages/
        image_set (str, optional): Dataset split to use.
            - Choose from "train", "val", or "trainval".
            Default is "train".
            "trainval" combines both the train and val sets.

    Returns:
        tuple:
            img (Tensor): Image tensor with shape [C, H, W].
            target (dict): 
                - "boxes": Tensor, shape = [num_objects, 4]
                  (xmin, ymin, xmax, ymax)
                - "labels": Tensor, shape = [num_objects]

    Example:
        >>> dataset = VOCObjectDetection(root="data/VOC2012", image_set="val")
        >>> print(len(dataset))
        5823
        >>> img, target = dataset[0]
        >>> print(img.shape)
        torch.Size([3, 375, 500])
        >>> print(target['boxes'])
        tensor([[48, 240, 195, 371]])
    """
    def __init__(self, root, image_set="train", img_size=640):
        # image_set = "train", "val", "trainval"
        self.image_set = image_set
        self.mosaic = MosaicAugmentor(img_size=img_size)
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

    # load image and annotation for mosaic augmentation
    def load_image_and_labels(self, idx):
        img = cv2.imread(self.images[idx])[:, :, ::-1]
        target = self.parse_annotation(self.annotations[idx])
        return img, target
    
    # Load bounding boxes and labels from image xml file
    def parse_annotation(self, annotation_path):
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
        targets = []
        # Iterate over all object elements in the XML
        for obj in root.findall("object"):
            # Save Labels
            label = obj.find("name").text
            label = VOC_CLASSES.index(obj.find("name").text)
            

            # Save Bounding Boxes
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)
            targets.append([xmin, ymin, xmax, ymax, 1.0, label])
        targets = np.array(targets)
        return targets
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Mosaic augmentation for train set
        img, target = self.load_image_and_labels(idx)
        if self.image_set == "train":
            img, target = self.mosaic(self, idx)
        else:
            img, target = self.mosaic.letterbox(img, target)
        img = MyTransform()(img)
        return img, torch.tensor(target, dtype=torch.float32)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    dataset = VOCObjectDetection(root="data/VOC2012", image_set="train")
    print(f"Number of samples: {len(dataset)}")
    img, target = dataset[1]
    m, n = img.max(), img.min()
    img = (img - n) / (m - n) * 255
    img_numpy = img.permute(1,2,0).numpy().astype(np.uint8)
    plt.imshow(img_numpy)
    plt.show()