import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import functools
from livelossplot import PlotLosses

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for images, masks, weights in tqdm.tqdm(iter(train_loader), "Training"):
        images, masks, weights = images.to(device), masks.to(device), weights.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        loss = criterion(outputs, masks, weights)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for images, masks, weights in tqdm.tqdm(iter(val_loader), "Validation"):
            images, masks, weights = images.to(device), masks.to(device), weights.to(device)
            
            outputs = model(images)
            
            loss = criterion(outputs, masks, weights)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=100, save_path='best_unet_model.pth'):
    plotlosses = PlotLosses()
    best_val_loss = float('inf')
    
    print(f'Using device: {device}')
    model.to(device)

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        val_loss = validate_epoch(model, val_loader, criterion, device)
        
        scheduler.step()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
    
        plotlosses.update({'loss': train_loss, 'val_loss': val_loss})
        plotlosses.send()
    
    return model
