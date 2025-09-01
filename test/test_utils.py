import torch
from tqdm import tqdm

def test_model(args, model, test_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        loop = tqdm(test_loader, leave=True)
        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            loss = torch.nn.BCELoss()(preds, masks)

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

    print(f"Test Loss: {total_loss/len(test_loader)}")
