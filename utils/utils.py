import cv2
import torch
import numpy as np
from PIL import Image
from torchvision.ops import nms

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]

def draw_bbox(img, target):
    # Convert img to numpy if PIL Image or tensor
    if isinstance(img, Image.Image):
        img = np.array(img)
    elif isinstance(img, torch.Tensor):
        img = (img * 255.0).permute(1, 2, 0).numpy().astype(np.uint8)

    result_img = img.copy()
    box_color = [tuple(np.random.randint(0, 256, size=3).tolist()) for _ in range(20)]

    for det in target:
        x1, y1, x2, y2, score, label = det
        label = int(label)
        pt1, pt2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(result_img, pt1, pt2, box_color[label], thickness=3)
        class_name = VOC_CLASSES[label]
        cv2.putText(result_img, class_name, pt1, cv2.FONT_HERSHEY_COMPLEX, 
                    0.5, box_color[label], thickness=1)
    
    return result_img

def decode_pred(pred, scale_idx, num_classes=20, img_size=640, conf_thres=0.25, iou_thres=0.45):
    """
    Decode YOLOv5n outputs and apply Non-Maximum Suppression (NMS) to filter predicted boxes.

    Args:
        pred (Tensor): Model output of shape (N, A * (5 + num_classes), H, W),  
            where N is the batch size, A is the number of anchors per scale,
            H and W are the feature map dimensions.
        scale_idx (int): Index of the detection scale.
            0 corresponds to small anchors, 1 to medium anchors, and 2 to large anchors.
        num_classes (int, optional): Number of classes in the dataset.
            Default is 20 for the VOC dataset.
        img_size (int, optional): Size of the input image. Default is 640.
        conf_thres (float, optional): Confidence score threshold for filtering low-confidence boxes.
            Default is 0.25.
        iou_thres (float, optional): IOU threshold for nms

    Returns:
        Tensor: A list of length N, where each element is a tensor of shape (num_boxes, 6).  
            Each row represents a detected box in the format [x1, y1, x2, y2, score, label].
    """
    A = 3
    N, _, H, W = pred.shape
    stride = img_size / H

    # Reshape prediction
    pred = pred.view(N, A, (5 + num_classes), H, W).permute(0, 1, 3, 4, 2)

    pred_obj = pred[..., 4]
    pred_txty = pred[..., :2]
    pred_twth = pred[..., 2:4]
    pred_cls = pred[..., 5:]

    # Anchors
    anchors = torch.tensor([
        [(10, 13), (16, 30), (33, 23)],
        [(30, 61), (62, 45), (59, 119)],
        [(116, 90), (156, 198), (373, 326)]
    ]) / img_size
    anchors = anchors[scale_idx]

    # Grid
    grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    grid = torch.stack([grid_x, grid_y], dim=-1).view(1, 1, H, W, 2).float().to(pred.device)

    # Decode boxes
    pred_txty = ((2.0 * torch.sigmoid(pred_txty) - 0.5 + grid) * stride)
    pred_twth = ((2.0 * torch.sigmoid(pred_twth)) ** 2) * anchors.view(1, A, 1, 1, 2)

    xy_min = pred_txty - pred_twth / 2
    xy_max = pred_txty + pred_twth / 2
    pred_boxes = torch.cat([xy_min, xy_max], dim=-1)  # (N, A, H, W, 4)

    # Flatten
    pred_boxes = pred_boxes.view(N, -1, 4)
    pred_obj = pred_obj.view(N, -1, 1)
    pred_cls = pred_cls.view(N, -1, num_classes)

    # Calculate scores
    class_probs = torch.sigmoid(pred_cls)
    class_scores, class_ids = torch.max(class_probs, dim=-1, keepdim=True)  # (N, A*H*W, 1)
    pred_scores = torch.sigmoid(pred_obj) * class_scores

    result = []

    for i in range(N):
        boxes, scores, labels = pred_boxes[i], pred_scores[i], class_ids[i]
        scores, labels = scores.squeeze(-1), labels.squeeze(-1)

        # Confidence filter
        mask = scores > conf_thres
        boxes, scores, labels = boxes[mask], scores[mask], labels[mask]

        if boxes.numel() == 0:
            result.append(torch.empty((0, 6), device=pred.device))
            continue

        # Apply NMS
        keep_idx = nms(boxes, scores, iou_thres)
        boxes, scores, labels = boxes[keep_idx], scores[keep_idx], labels[keep_idx]

        # Final format: [x1, y1, x2, y2, score, label]
        detections = torch.cat([boxes, scores.unsqueeze(-1), labels.unsqueeze(-1)], dim=1)
        result.append(detections)

    return result

def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, 0)  # (N, C, H, W)

    # Pad targets to have same number of boxes per batch
    max_boxes = max(t.shape[0] for t in targets)
    padded_targets = []
    for t in targets:
        if t.shape[0] < max_boxes:
            pad = torch.zeros((max_boxes - t.shape[0], 6), dtype=t.dtype)
            t = torch.cat([t, pad], dim=0)
        padded_targets.append(t)

    targets = torch.stack(padded_targets, 0)  # (N, B, 6)
    return images, targets