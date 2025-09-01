import torch
import matplotlib.pyplot as plt

def save_checkpoint(model, optimizer, filename="checkpoint.pth.tar"):
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def load_checkpoint(checkpoint_file, model, optimizer):
    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

def plot_predictions(model, loader, device, num_examples=3):
    model.eval()
    for idx, (images, masks) in enumerate(loader):
        images = images.to(device)
        preds = model(images)
        preds = (preds > 0.5).float()
        plt.figure(figsize=(10,5))
        plt.subplot(1,3,1)
        plt.imshow(images[0].permute(1,2,0).cpu())
        plt.title("Image")
        plt.subplot(1,3,2)
        plt.imshow(masks[0].squeeze().cpu(), cmap="gray")
        plt.title("Ground Truth")
        plt.subplot(1,3,3)
        plt.imshow(preds[0].squeeze().cpu(), cmap="gray")
        plt.title("Prediction")
        plt.show()
        if idx+1 >= num_examples:
            break
