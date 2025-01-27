import torch
import torch.nn.functional as F

def one_hot_encode(targets, num_classes):
    targets = targets.unsqueeze(1) 
    one_hot = torch.zeros(targets.size(0), num_classes, *targets.shape[2:], device=targets.device)
    one_hot.scatter_(1, targets, 1)
    return one_hot

def dice_coefficient(preds, targets, smooth=1e-6):
    preds_flat = preds.contiguous().view(-1)
    targets_flat = targets.contiguous().view(-1)
    
    intersection = (preds_flat * targets_flat).sum()
    dice = (2. * intersection + smooth) / (preds_flat.sum() + targets_flat.sum() + smooth)
    return dice

def dice_loss(preds, targets, smooth=1e-6):
    dice = dice_coefficient(preds, targets, smooth)
    return 1 - dice

def combined_dice_cross_entropy_loss(preds, targets, num_classes=4, smooth=1e-6):
    targets_one_hot = one_hot_encode(targets, num_classes)
    preds_softmax = F.softmax(preds, dim=1)

    dice_losses = []
    for i in range(num_classes):
        dice_losses.append(dice_loss(preds_softmax[:, i], targets_one_hot[:, i], smooth))
    mean_dice_loss = torch.mean(torch.stack(dice_losses))
    ce_loss = F.cross_entropy(preds, targets.squeeze(1).long())

    combined_loss = mean_dice_loss + ce_loss
    return combined_loss
