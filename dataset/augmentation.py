import random
import cv2
import numpy as np
from PIL import Image

class MosaicAugmentor:
    def __init__(self, img_size=640, mosaic_prob=1.0):
        self.img_size = img_size
        self.mosaic_prob = mosaic_prob

    def __call__(self, dataset, idx):
        """ 
        dataset: VOCObjectDetection 객체
        idx: 현재 index
        """
        # 114 padded images and labels with probability 1 - mosaic_prob
        if random.random() > self.mosaic_prob:
            img, target = dataset.load_image_and_lagels(idx)
            img, target = self.letterbox(img, target)
            return img, target
        # Generate augmented image with probability mosaic_prob
        # Sample 3 random images for mosaic augmentation
        indices = [idx] + random.choices(range(len(dataset)), k=3)

        # Create result image with size (640, 640)
        s = self.img_size
        yc, xc = s // 2, s // 2  # Center of the new image
        mosaic_img = np.full((s, s, 3), 114, dtype=np.uint8)  # 114 Color for Background

        # New bboxes for new image
        mosaic_boxes = []
        mosaic_labels = []

        for i, index in enumerate(indices):
            img, target = dataset.load_image_and_lagels(index)
            h, w = img.shape[:2]

            # Radnom Scale Resize
            scale = random.uniform(0.4, 1.0)
            new_w, new_h = int(w * scale), int(h * scale)
            img = np.array(Image.fromarray(img).resize((new_w, new_h)))

            # Calculate 4 image position
            if i == 0:  # top-left
                # Position for new image
                x1a, y1a, x2a, y2a = max(xc - new_w, 0), max(yc - new_h, 0), xc, yc
                # Position for random scaled original image
                # Cut from bottom-right
                x1b, y1b, x2b, y2b = new_w - (x2a - x1a), new_h - (y2a - y1a), new_w, new_h
            elif i == 1:  # top-right
                x1a, y1a, x2a, y2a = xc, max(yc - new_h, 0), min(xc + new_w, s), yc
                # Cut from bottom-left
                x1b, y1b, x2b, y2b = 0, new_h - (y2a - y1a), min(new_w, x2a - x1a), new_h
            elif i == 2:  # bottom-left
                x1a, y1a, x2a, y2a = max(xc - new_w, 0), yc, xc, min(s, yc + new_h)
                # Cut from top-right
                x1b, y1b, x2b, y2b = new_w - (x2a - x1a), 0, new_w, min(y2a - y1a, new_h)
            else:  # bottom-right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + new_w, s), min(s, yc + new_h)
                # Cut from top-left
                x1b, y1b, x2b, y2b = 0, 0, min(new_w, x2a - x1a), min(new_h, y2a - y1a)

            # Attach original image to new image
            mosaic_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]

            # Bounding Box 좌표 이동
            boxes = target["boxes"].copy()
            labels = target["labels"].copy()
            if len(boxes) > 0:
                for i, box in enumerate(boxes):
                    box[[0, 2]] = box[[0, 2]] * (x2b - x1b) / w + x1a
                    box[[1, 3]] = box[[1, 3]] * (y2b - y1b) / h + y1a
                    x_box = box[2] - box[0]
                    y_box = box[3] - box[1]
                    if (x_box > 4) & (y_box > 4):
                        mosaic_boxes.append(box)
                        mosaic_labels.append(labels[i])

        target["boxes"] = mosaic_img
        target["labels"] = mosaic_labels
        # 최종 target
        return mosaic_img, target

    def letterbox(self, img, target, new_shape=(640, 640), color=(114, 114, 114)):
        h, w = img.shape[:2]
        scale = min(new_shape[0] / h, new_shape[1] / w)
        new_w, new_h = int(w * scale), int(h * scale)

        # resize
        img_resized = np.array(Image.fromarray(img).resize((new_w, new_h)))

        # padding 계산
        dw = new_shape[1] - new_w # width padding
        dh = new_shape[0] - new_h # height padding

        top, bottom = dh // 2, dh - dh // 2 # Top, Bottom padding size
        left, right = dw // 2, dw - dw // 2 # Left, Right padding size

        # Apply padding
        img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right,
                                        cv2.BORDER_CONSTANT, value=color)

        # Resize and move bounding box
        boxes = target["boxes"].copy()
        if len(boxes) > 0:
            for i, box in enumerate(boxes):
                box[[0, 2]] = box[[0, 2]] * scale + left
                box[[1, 3]] = box[[1, 3]] * scale + top
        target["boxes"] = boxes

        return img_padded, target