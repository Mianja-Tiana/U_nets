import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class UNetLoss(nn.Module):
    def __init__(self):
        super(UNetLoss, self).__init__()

    def forward(self, outputs, targets, weights=None):

        outputs = F.softmax(outputs, dim=1)
        outputs = outputs.view(*outputs.size()[:2], -1)
        targets = targets.view(*targets.size()[:1], -1)

        loss = F.cross_entropy(outputs, targets, reduction="none")
        
        if weights is not None:
            loss *= weights.view(*weights.size()[:1], -1)

        loss = loss.sum(axis=0).mean()
        
        return loss