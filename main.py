from config import flags
import time, datetime
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from model import YOLOv5n
from dataset import VOCObjectDetection
from train import Yolo_Loss
from utils import collate_fn
from tqdm import tqdm


def main():
    start_time = time.time()
    
    # Load dataset
    train_dataset = VOCObjectDetection(root=flags["data_dir"], image_set='train')
    train_loader = DataLoader(train_dataset, batch_size=flags["batch_size"], shuffle=True, collate_fn=collate_fn)

    # Load model
    model = YOLOv5n(num_classes=20, anchor=3)

    # Loss function and optimizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    criterion = Yolo_Loss(num_classes=flags["num_classes"], img_size=320, device=device)
    optimizer = Adam(params=model.parameters(), lr = flags["learning_rate"])

    train_loss = []
    for epoch in range(flags["num_epochs"]):
        epoch_loss = 0
        for (data, target) in tqdm(train_loader):
            preds = model(data)

            loss, loss_items = criterion(preds, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.cpu().item())
            epoch_loss += loss.cpu().item()
        print(f"[Epoch {epoch+1}/{flags['num_epochs']}],",
            f"Total Loss: {epoch_loss:.4f},",
            f"(box: {loss_items['box']:.4f},",
            f"obj: {loss_items['obj']:.4f},",
            f"cls: {loss_items['cls']:.4f})")


    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Main completed in: {str(datetime.timedelta(seconds=elapsed_time))}")

if __name__ == "__main__":
    main()
