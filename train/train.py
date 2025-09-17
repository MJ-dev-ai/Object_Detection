from tqdm import tqdm
import torch, os

def train_model(model, train_loader, criterion, optimizer, flags, device):
    model = model.to(device)
    model.train()
    train_loss = []
    for epoch in range(flags["num_epochs"]):
        epoch_loss = 0
        for (data, target) in tqdm(train_loader):
            data, target = data.to(device), target.to(device)

            preds = model(data)

            loss, loss_items = criterion(preds, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.cpu().item())
            epoch_loss += loss.cpu().item()
        epoch_loss /= len(train_loader)
        print(f"[Epoch {epoch+1}/{flags['num_epochs']}],",
            f"Total Loss: {epoch_loss:.4f},",
            f"(box: {loss_items['box']:.4f},",
            f"obj: {loss_items['obj']:.4f},",
            f"cls: {loss_items['cls']:.4f})")
        if (epoch + 1) % 10 == 0:
            if not os.path.exists(flags["model_dir"]): os.mkdir(flags["model_dir"])
            filename = "parameters.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(flags["model_dir"], filename))