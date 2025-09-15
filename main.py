from config import flags
import time, datetime
import torch
from model import YOLOv5n
from dataset import VOCObjectDetection
from train import Yolo_Loss


def main():
    start_time = time.time()
    
    # 가짜 데이터
    batch_size = 2
    A = 3
    H, W = 20, 20
    num_classes = 20
    img_size = 640

    # 랜덤 target (중심좌표, w, h 포함)
    target = {
        "boxes": torch.tensor([
            [[100, 150, 200, 250], [300, 300, 400, 400]],  # 첫 번째 이미지의 2개 박스
            [[50, 50, 100, 100], [250, 250, 350, 350]]     # 두 번째 이미지의 2개 박스
        ]),
        "labels": torch.tensor([
            [0, 5],
            [3, 7]
        ])
    }

    loss_fn = Yolo_Loss(num_classes=num_classes, img_size=img_size)

    # 가짜 predictions: target과 완전히 동일하게 설정
    preds = torch.zeros(batch_size, A*(5+num_classes), H, W)
    encoded_target = loss_fn.build_target(target, scale_idx=2, device='cpu')

    # preds의 box와 obj, cls를 target과 동일하게 맞춰줌
    preds = encoded_target.clone()

    # forward 실행
    loss = loss_fn(preds, target, scale_idx=2)
    print("Loss (Pred == Target):", loss.item())

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Main completed in: {str(datetime.timedelta(seconds=elapsed_time))}")

if __name__ == "__main__":
    main()
