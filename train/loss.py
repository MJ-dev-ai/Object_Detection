import torch
from torch import nn



class Yolo_Loss(nn.Module):
    def __init__(self, anchors=None, num_classes=20, img_size=640, lambda_box=0.05, lambda_obj=1.0, lambda_cls=0.5, device='cpu'):
        super(Yolo_Loss, self).__init__()
        self.num_classes=num_classes
        self.img_size=img_size
        self.anchors = anchors
        if anchors is None:
            self.anchors = torch.tensor([
                (10, 13), (16, 30), (33, 23),
                (30, 61), (62, 45), (59, 119),
                (116, 90), (156, 198), (373, 326)
            ]) / self.img_size # Normalize with image size
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.lambda_cls = lambda_cls
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')
        self.device = device

    def build_target(self, targets, scale_idx, num_classes=20, device='cpu'):
        """
        targets: (N, M, 5) -> [x1, y1, x2, y2, class_id]
        N: batch size
        M: Number of boxes for each images
        """
        H, W = [(80, 80), (40, 40), (20, 20)][scale_idx]
        boxes, labels = targets["boxes"], targets["labels"]
        if labels.ndim == 2:
            labels = labels.unsqueeze(-1)
        target = torch.cat([boxes.float(), labels.float()], dim=-1)
        anchors = self.anchors[scale_idx * 3:(scale_idx + 1) * 3]

        N = target.shape[0]
        A = anchors.shape[0]

        target_tensor = torch.zeros((N, A, 5 + num_classes, H, W), device=device)

        for b in range(N): # Number of data
            for box in target[b]:
                x1, y1, x2, y2, cls = box
                if x2 <= x1 or y2 <= y1:
                    continue  # invalid box

                # Normalize coordinates and size of bbox into [0, 1]
                cx = (x1 + x2) / 2 / self.img_size
                cy = (y1 + y2) / 2 / self.img_size
                w = (x2 - x1) / self.img_size
                h = (y2 - y1) / self.img_size

                gx, gy = int(cx * W), int(cy * H) # Points in (H, W)
                dx, dy = cx * W - gx, cy * H - gy # Position of anchor from gx, gy

                # Choose best anchor
                ratios = torch.stack([w / anchors[:, 0], h / anchors[:, 1]], dim=1) # Ratio of width and height
                ious = torch.min(ratios, 1 / ratios).prod(1) # 
                best_anchor = torch.argmax(ious) # Choose most similar size anchor as target box

                # Save target
                target_tensor[b, best_anchor, 0, gy, gx] = dx
                target_tensor[b, best_anchor, 1, gy, gx] = dy
                target_tensor[b, best_anchor, 2, gy, gx] = torch.log(w / anchors[best_anchor][0] + 1e-6)
                target_tensor[b, best_anchor, 3, gy, gx] = torch.log(h / anchors[best_anchor][1] + 1e-6)
                target_tensor[b, best_anchor, 4, gy, gx] = 1.0  # objectness
                target_tensor[b, best_anchor, 5 + int(cls), gy, gx] = 1.0  # one-hot class

        return target_tensor

    def forward(self, preds, target, scale_idx):
        H, W = [(80, 80), (40, 40), (20, 20)][scale_idx]
        stride = self.img_size / H
        A = 3
        anchors = self.anchors[scale_idx * A:(scale_idx + 1) * A]
        target = self.build_target(target, scale_idx = scale_idx, num_classes=self.num_classes, device=self.device)
        preds = preds.view(-1, A, 5 + self.num_classes, H, W)
        preds_box = preds[:, :, :4, :, :]
        preds_obj = preds[:, :, 4, :, :]
        preds_cls = preds[:, :, 5:, :, :]
        
        target_box = target[:, :, :4, :, :]
        target_obj = target[:, :, 4, :, :]
        target_cls = target[:, :, 5:, :, :]
        
        # Decode box coordinate
        # xgrid, ygrid: tensor(H, W) -> torch.stack((xgrid, ygrid), 2): tensor(H, W, 2)
        ygrid, xgrid = torch.meshgrid([torch.arange(H, device=self.device), torch.arange(W, device=self.device)], indexing='ij')
        grid = torch.stack((xgrid, ygrid), 2).view(1, 1, H, W, 2).float()
        pred_tx_ty = preds_box[:, :, :2, :, :].permute(0, 1, 3, 4, 2) # Permute to (N, A, H, W, 2) to fit in grid
        pred_tw_th = preds_box[:, :, 2:, :, :].permute(0, 1, 3, 4, 2)
        pred_xy = (2.0 * torch.sigmoid(pred_tx_ty) - 0.5 + grid) * stride
        pred_wh = ((2.0 * torch.sigmoid(pred_tw_th)) ** 2) * anchors.view(1, A, 1, 1, 2) # (0, 4) size wh * anchor
        preds_box_decode = torch.cat([pred_xy, pred_wh], dim=-1) # (N, A, H, W, 4)
        preds_box_decode = preds_box_decode[preds_obj==1]
        
        target_tx_ty = target_box[:, :, :2, :, :].permute(0, 1, 3, 4, 2)
        target_tw_th = target_box[:, :, 2:, :, :].permute(0, 1, 3, 4, 2)
        target_xy = (target_tx_ty + grid) * stride
        target_wh = torch.exp(target_tw_th) * anchors.view(1, A, 1, 1, 2)
        target_box_decode = torch.cat([target_xy, target_wh], dim=-1) # (N, A, H, W, 4)
        target_box_decode = target_box_decode[target_obj==1]

        iou = self.bbox_iou(preds_box_decode.view(-1,4), target_box_decode.view(-1,4)).sum()

        loss_box = 1.0 - iou
        loss_obj = self.bce(preds_obj, target_obj)
        loss_cls = self.bce(preds_cls, target_cls)
        return (self.lambda_box * loss_box + self.lambda_obj * loss_obj + self.lambda_cls * loss_cls)
    
    def bbox_iou(self, box1, box2, eps=1e-7):
        # x1, x2 = cx - w/2, cx + w/2
        # y1, y2 = cy - h/2, cy + h/2
        b1_x1 = box1[:, 0] - box1[:, 2] / 2
        b1_y1 = box1[:, 1] - box1[:, 3] / 2
        b1_x2 = box1[:, 0] + box1[:, 2] / 2
        b1_y2 = box1[:, 1] + box1[:, 3] / 2

        b2_x1 = box2[:, 0] - box2[:, 2] / 2
        b2_y1 = box2[:, 1] - box2[:, 3] / 2
        b2_x2 = box2[:, 0] + box2[:, 2] / 2
        b2_y2 = box2[:, 1] + box2[:, 3] / 2

        # Calculate Area of intersection of two boxes
        inter_x1 = torch.max(b1_x1, b2_x1)
        inter_y1 = torch.max(b1_y1, b2_y1)
        inter_x2 = torch.min(b1_x2, b2_x2)
        inter_y2 = torch.min(b1_y2, b2_y2)
        inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

        # Union
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        # B1 ∪ B2 = B1 + B2 - (B1 ∩ B2)
        union_area = b1_area + b2_area - inter_area + eps

        # IOU = (B1 ∩ B2) / (B1 ∪ B2)
        return inter_area / union_area