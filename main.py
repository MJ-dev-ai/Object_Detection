from config import flags
import time, datetime
import torch
import os
from torch.optim import Adam
from torch.utils.data import DataLoader
from model import YOLOv5n
from dataset import VOCObjectDetection
from train import Yolo_Loss
from utils import collate_fn
from train import train_model

def main():
    start_time = time.time()
    
    # Load dataset
    train_dataset = VOCObjectDetection(root=flags["data_dir"], image_set='train', img_size=320)
    train_loader = DataLoader(train_dataset, batch_size=flags["batch_size"], shuffle=True, collate_fn=collate_fn, num_workers=flags["num_workers"])

    # Load model
    model = YOLOv5n(num_classes=20, anchor=3)

    # Loss function and optimizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    criterion = Yolo_Loss(num_classes=flags["num_classes"], img_size=320, device=device)
    optimizer = Adam(params=model.parameters(), lr = flags["learning_rate"])
    if os.path.exists(os.path.join(flags["model_dir"], "parameters.pt")):
        checkpoint = torch.load(os.path.join(flags["model_dir"], "parameters.pt"), map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        train_model(model, train_loader, criterion, optimizer, flags, device)
    
    # Validate
    val_dataset = VOCObjectDetection(root=flags["data_dir"], image_set='val', img_size=320)
    from utils import validate
    img_pred, img_target = validate(model, val_dataset, device, img_size=320)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img_target)
    plt.title("Ground Truth")
    plt.subplot(1, 2, 2)
    plt.imshow(img_pred)
    plt.title("Prediction")
    plt.show()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Main completed in: {str(datetime.timedelta(seconds=elapsed_time))}")

if __name__ == "__main__":
    main()
