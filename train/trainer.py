import torch
from tqdm import tqdm
from train.train_utils import combined_loss

def train_model(args, model, train_loader, optimizer, device):
    model.train()
    for epoch in range(args.epochs):
        loop = tqdm(train_loader, leave=True)
        epoch_loss = 0
        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            loss = combined_loss(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            loop.set_description(f"Epoch [{epoch+1}/{args.epochs}]")
            loop.set_postfix(loss=loss.item())

        print(f"Epoch {epoch+1} Loss: {epoch_loss/len(train_loader)}")
    return model
