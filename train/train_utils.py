import torch
import torch.nn as nn

def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)
    loss = (1 - ((2. * intersection + smooth) /
                (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
    return loss.mean()

def combined_loss(pred, target):
    bce = nn.BCELoss()(pred, target)
    dsc = dice_loss(pred, target)
    return bce + dsc
