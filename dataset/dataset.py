from torch.utils.data import Dataset
import os
from xml.etree import ElementTree as ET
from PIL import Image
import numpy as np
from augmentation import MosaicAugmentor

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
    """
    PASCAL VOC Object Detection Dataset Loader.

    이 클래스는 PASCAL VOC 2012 데이터셋을 로드하여 torch.utils.data의 Dataset 형식으로 제공합니다.
    - 이미지와 XML 어노테이션 파일을 연결하여 `(image, target)` 튜플로 반환합니다.
    - `target`은 dictionary로, "boxes"와 "labels"를 포함합니다.
    - "boxes"는 객체의 bounding box 좌표를, "labels"는 객체의 클래스 인덱스를 나타냅니다.

    Args:
        root (str): VOC 데이터셋의 최상위 디렉토리 경로.
            예시:
            root/
                Annotations/
                ImageSets/
                JPEGImages/
        image_set (str, optional): 사용할 데이터셋 split.
            - "train", "val", "trainval" 중 하나를 선택.
            기본값은 "train".
            trainval은 train과 val을 합친 데이터셋.

    Returns:
        tuple:
            img (Tensor): [C, H, W] 크기의 이미지 텐서.
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
    def __init__(self, root, image_set="train"):
        # image_set = "train", "val", "trainval"
        self.image_set = image_set
        self.mosaic = MosaicAugmentor(img_size=640)
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
    def load_image_and_lagels(self, idx):
        img = np.array(Image.open(self.images[idx]).convert("RGB"))
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
        boxes, labels = np.array(boxes), np.array(labels)
        target = {"boxes": boxes, "labels": labels}
        return target
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Mosaic augmentation for train3 set
        
        img, target = self.load_image_and_lagels(idx)
        # Normalize image and convert data into tensor
        
        if self.image_set == "train":
            # MosaicAugmentor needs dataset for argv
            img, target = self.mosaic(self, idx)
        # Resize and pad inputs
        else:
            img, target = self.mosaic.letterbox(img, target)

        img, boxes, labels = MyTransform()(img, target["boxes"], target["labels"])
        target = {"boxes": boxes, "labels": labels}

        return img, target
    
if __name__ == "__main__":
    dataset = VOCObjectDetection(root="data/VOC2012", image_set="val")
    print(f"Number of samples: {len(dataset)}")
    img, target = dataset[0]
    print(f"Image shape: {img.shape}")
    print(f"Boxes: {target['boxes']}")
    print(f"Labels: {target['labels']}")