import random
import cv2
import numpy as np
from PIL import Image

class MosaicAugmentor:
    """
    Mosaic Augmentator for YOLOv5n
    """
    def __init__(self, img_size=640, mosaic_prob=1.0):
        self.img_size = img_size
        self.mosaic_prob = mosaic_prob

    def __call__(self, dataset, idx):
        """
        dataset: VOCObjectDetection object
        idx: current sample index
        """
        # Probability not to apply mosaic augmentation
        if random.random() > self.mosaic_prob:
            img, target = dataset.load_image_and_lagels(idx)
            img, target = self.letterbox(img, target)
            return img, target

        # Select 4 images for Mosaic (including current index)
        indices = [idx] + random.choices(range(len(dataset)), k=3)

        s = self.img_size
        yc, xc = s // 2, s // 2  # center coordinates
        mosaic_img = np.full((s, s, 3), 114, dtype=np.uint8)  # empty background

        mosaic_targets = []

        for i, index in enumerate(indices):
            img, target = dataset.load_image_and_labels(index)
            h, w = img.shape[:2]

            # Random scaling
            scale = random.uniform(0.4, 1.0)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

            # Calculate placement position
            if i == 0:  # top-left
                x1a, y1a, x2a, y2a = max(xc - new_w, 0), max(yc - new_h, 0), xc, yc
                x1b, y1b, x2b, y2b = new_w - (x2a - x1a), new_h - (y2a - y1a), new_w, new_h
                pos_x, pos_y = xc - new_w, yc - new_h
            elif i == 1:  # top-right
                x1a, y1a, x2a, y2a = xc, max(yc - new_h, 0), min(xc + new_w, s), yc
                x1b, y1b, x2b, y2b = 0, new_h - (y2a - y1a), min(new_w, x2a - x1a), new_h
                pos_x, pos_y = xc, yc - new_h
            elif i == 2:  # bottom-left
                x1a, y1a, x2a, y2a = max(xc - new_w, 0), yc, xc, min(s, yc + new_h)
                x1b, y1b, x2b, y2b = new_w - (x2a - x1a), 0, new_w, min(new_h, y2a - y1a)
                pos_x, pos_y = xc - new_w, yc
            else:  # bottom-right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + new_w, s), min(s, yc + new_h)
                x1b, y1b, x2b, y2b = 0, 0, min(new_w, x2a - x1a), min(new_h, y2a - y1a)
                pos_x, pos_y = xc, yc

            # Combine image
            mosaic_img[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]

            # Move bounding boxes
            if target.shape[0] > 0:
                # Coordinate transform + clip
                target[:, [0, 2]] = np.clip(target[:, [0, 2]] * (x2a - x1a) / w + pos_x, 0, s - 1)
                target[:, [1, 3]] = np.clip(target[:, [1, 3]] * (y2a - y1a) / h + pos_y, 0, s - 1)

                # Filter out very small boxes
                wh = target[:, 2:4] - target[:, 0:2]  # (N, 2)
                mask = (wh[:, 0] > 4) & (wh[:, 1] > 4)
                target = target[mask]

                mosaic_targets.append(target)

        # Merge targets
        if len(mosaic_targets) > 0:
            mosaic_targets = np.concatenate(mosaic_targets, axis=0).astype(np.float32)  # (N,6)
        else:
            mosaic_targets = np.zeros((0, 6), dtype=np.float32)

        return mosaic_img, mosaic_targets

    def letterbox(self, img, target, new_shape=(640, 640), color=(114, 114, 114)):
        """
        Resize image and boxes with padding
        """
        h, w = img.shape[:2]
        scale = min(new_shape[0] / h, new_shape[1] / w)
        new_w, new_h = int(w * scale), int(h * scale)

        # resize
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        # calculate padding
        dw = new_shape[1] - new_w  # width padding
        dh = new_shape[0] - new_h  # height padding
        top, bottom = dh // 2, dh - dh // 2
        left, right = dw // 2, dw - dw // 2

        # apply padding
        img_padded = cv2.copyMakeBorder(
            img_resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=color
        )

        # move bounding boxes
        if target.shape[0] > 0:
            target[:, [0, 2]] = target[:, [0, 2]] * scale + left
            target[:, [1, 3]] = target[:, [1, 3]] * scale + top

        return img_padded, target