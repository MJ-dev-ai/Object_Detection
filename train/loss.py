import torch
from torch import nn

class Yolo_Loss(nn.Module):
    def __init__(self, anchors=None, num_classes=20, img_size=640,
                 lambda_box=0.05, lambda_obj=1.0, lambda_cls=0.5, device='cpu'):
        super(Yolo_Loss, self).__init__()
        self.device = device
        self.num_classes = num_classes
        self.img_size = img_size
        self.anchors = anchors
        if anchors is None:
            self.anchors = torch.tensor([
                [(10, 13), (16, 30), (33, 23)],
                [(30, 61), (62, 45), (59, 119)],
                [(116, 90), (156, 198), (373, 326)]
            ]) / self.img_size  # normalize
        self.anchors = self.anchors.to(device)
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.lambda_cls = lambda_cls
        self.bce_cls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=self.device))
        self.bce_obj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([1.0], device=self.device))

    def build_target(self, targets, scale_idx, num_classes=20):
        """
        Argv:
            targets: tensor with lists
                    list of Tensors [(x1, y1, x2, y2, obj, cls), ...]
            scale_idx (int): 0, 1 or 2
                            0 # small anchor
                            1 # medium anchor
                            2 # large anchor
        Returns:
        target_tensor: (N, A, 5+num_classes, H, W)
        """
        stride = [8, 16, 32][scale_idx]
        H, W = int(self.img_size / stride), int(self.img_size / stride)
        N = len(targets)
        A = 3
        device = self.device
        anchors = self.anchors[scale_idx].to(device).float()
        target_tensor = torch.zeros((N, A, 5 + num_classes, H, W), device=device, dtype=torch.float32)

        for b in range(N):
            target = targets[b].to(device)
            if target.numel() == 0:
                continue

            for t in target:  # each box: [x1, y1, x2, y2, obj, class_id]
                x1, y1, x2, y2, obj, cls = t
                if x2 <= x1 or y2 <= y1:  # skip invalid boxes
                    continue

                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                bw = x2 - x1
                bh = y2 - y1

                gx, gy = int((cx / stride).clamp(0, W - 1).item()), int((cy / stride).clamp(0, H - 1).item())
                dx, dy = (cx / stride - gx), (cy /stride - gy)

                # Select best anchor
                wh_ratio = torch.stack([bw / anchors[:, 0], bh / anchors[:, 1]], dim=1)
                with torch.no_grad():
                    inv = 1.0 / (wh_ratio + 1e-6)
                    mins = torch.min(wh_ratio, inv)
                    iou_like = mins.prod(dim=1)
                    best_anchor = int(torch.argmax(iou_like).item())

                # Write to target tensor
                # YOLOv5의 tx, ty는 2*sigmoid(tx)-0.5로 복원되므로, target에서는 dx+0.5를 그대로 저장
                target_tensor[b, best_anchor, 0, gy, gx] = dx + 0.5
                target_tensor[b, best_anchor, 1, gy, gx] = dy + 0.5
                target_tensor[b, best_anchor, 2, gy, gx] = bw / anchors[best_anchor, 0]
                target_tensor[b, best_anchor, 3, gy, gx] = bh / anchors[best_anchor, 1]
                target_tensor[b, best_anchor, 4, gy, gx] = 1.0  # objectness
                cls_idx = int(cls.item())
                if 0 <= cls_idx < num_classes:
                    target_tensor[b, best_anchor, 5 + cls_idx, gy, gx] = 1.0

        return target_tensor

    def forward(self, preds, targets):
        """
        preds: list of 3 tensors
            [
                [N, 3*(5+num_classes), 80, 80],  # small scale
                [N, 3*(5+num_classes), 40, 40],  # medium scale
                [N, 3*(5+num_classes), 20, 20]   # large scale
            ]
        target: tensor list
            [x1, y1, x2, y2, obj, cls]
        """
        assert isinstance(preds, (list, tuple)) and len(preds) == 3, \
            "YOLOv5n must output 3 feature maps."

        total_loss = torch.tensor(0.0, device=self.device)
        loss_items = {'box': 0, 'obj': 0, 'cls': 0}

        for scale_idx, pred in enumerate(preds):
            loss, l_box, l_obj, l_cls = self.compute_loss_per_scale(pred, targets, scale_idx)
            total_loss += loss
            loss_items['box'] += l_box.item()
            loss_items['obj'] += l_obj.item()
            loss_items['cls'] += l_cls.item()

        return total_loss, loss_items
    
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

        inter_x1 = torch.max(b1_x1, b2_x1)
        inter_y1 = torch.max(b1_y1, b2_y1)
        inter_x2 = torch.min(b1_x2, b2_x2)
        inter_y2 = torch.min(b1_y2, b2_y2)
        inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        union_area = b1_area + b2_area - inter_area + eps

        return inter_area / union_area
    
    def compute_loss_per_scale(self, pred, target, scale_idx):
        """
        pred: [N, 3*(5+num_classes), H, W]
        target: list of length N, Each element is tensor [x1, y1, x2, y2, obj, cls]
        """
        pred = pred.to(self.device)
        N = pred.shape[0]
        stride = [8, 16, 32][scale_idx]
        H, W = int(self.img_size / stride), int(self.img_size / stride)
        A = 3

        anchors = self.anchors[scale_idx].to(self.device).float()


        target_tensor = self.build_target(target, scale_idx, self.num_classes)

        pred = pred.view(N, A, 5 + self.num_classes, H, W)
        pred_box = pred[:, :, :4, :, :]
        pred_obj = pred[:, :, 4, :, :]
        pred_cls = pred[:, :, 5:, :, :]

        target_box = target_tensor[:, :, :4, :, :]
        target_obj = target_tensor[:, :, 4, :, :]
        target_cls = target_tensor[:, :, 5:, :, :]

        # Create grid
        ygrid, xgrid = torch.meshgrid(
            [torch.arange(H, device=self.device),
             torch.arange(W, device=self.device)],
             indexing='ij'
        )
        grid = torch.stack((xgrid, ygrid), 2).view(1, 1, H, W, 2).float()

        # Decode predictions
        pred_tx_ty = pred_box[:, :, 0:2, :, :].permute(0, 1, 3, 4, 2)
        pred_tw_th = pred_box[:, :, 2:4, :, :].permute(0, 1, 3, 4, 2)

        pred_xy = (2.0 * torch.sigmoid(pred_tx_ty) - 0.5 + grid) * stride
        pred_wh = ((2.0 * torch.sigmoid(pred_tw_th)) ** 2) * anchors.view(1, A, 1, 1, 2) * stride
        pred_box_decode = torch.cat([pred_xy, pred_wh], dim=-1)

        # Decode targets
        target_tx_ty = target_box[:, :, 0:2, :, :].permute(0, 1, 3, 4, 2)
        target_tw_th = target_box[:, :, 2:4, :, :].permute(0, 1, 3, 4, 2)

        target_xy = (target_tx_ty + grid) * stride
        target_wh = (target_tw_th ** 2) * anchors.view(1, A, 1, 1, 2)
        target_box_decode = torch.cat([target_xy, target_wh], dim=-1)

        # Positive mask
        pos_mask = (target_obj == 1)
        pos_idx = pos_mask.view(-1)

        pred_pos = pred_box_decode.view(-1, 4)[pos_idx]
        target_pos = target_box_decode.view(-1, 4)[pos_idx]

        # IoU loss
        if pred_pos.numel() == 0:
            loss_box = torch.tensor(0.0, device=self.device)
        else:
            ious = self.bbox_iou(pred_pos, target_pos)
            loss_box = (1.0 - ious).mean()

        # Objectness loss
        loss_obj = self.bce_obj(pred_obj, target_obj)

        # Classification loss
        pred_cls_permute = pred_cls.permute(0, 1, 3, 4, 2)
        target_cls_permute = target_cls.permute(0, 1, 3, 4, 2)
        if pos_mask.sum() > 0:
            loss_cls = self.bce_cls(pred_cls_permute, target_cls_permute)
        else:
            loss_cls = torch.tensor(0.0, device=self.device)

        total_loss = (
            self.lambda_box * loss_box +
            self.lambda_obj * loss_obj +
            self.lambda_cls * loss_cls
        )
        return total_loss, loss_box.detach(), loss_obj.detach(), loss_cls.detach()