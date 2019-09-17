import torch.nn as nn


class UNetLoss(nn.Module):

    def __init__(self):
        super(UNetLoss, self).__init__()
        self.smooth = 1.0

    def dice_loss(self, y_pred, y_true, metrics):
        loss = _dice_loss(y_pred, y_true, self.smooth)
        metrics['dice_loss'] += loss.data.cpu().numpy()
        return loss

    def bce_loss(self, y_pred, y_true, metrics):
        loss = _bce_loss(y_pred, y_true, self.smooth)
        metrics['bce_loss'] += loss.data.cpu().numpy()
        return loss

    def bce_dice_loss(self, y_pred, y_true, metrics):
        bce_weight = 0.5

        dice = self.dice_loss(y_pred, y_true, metrics)
        bce = self.bce_loss(y_pred, y_true, metrics)
        loss = bce_weight * bce + (1. - bce_weight) * dice
        metrics['bce_dice_loss'] = loss.data.cpu().numpy()

        return loss

    def forward(self, y_pred, y_true, metrics):
        assert y_pred.size() == y_true.size()

        loss = self.bce_dice_loss(y_pred, y_true, metrics)
        metrics['loss'] = metrics['bce_dice_loss']

        return loss

def _dice_loss(y_pred, y_true, smooth):
    y_pred = y_pred.contiguous()
    y_true = y_true.contiguous()
    intersection = (y_pred * y_true).sum(dim=2).sum(dim=2)
    dice_loss = (1 - ((2. * intersection + smooth) / (y_pred.sum(dim=2).sum(dim=2) + y_true.sum(dim=2).sum(dim=2) + smooth)))
    dice_loss = dice_loss.mean()
    return dice_loss

def _bce_loss(y_pred, y_true, smooth):
    bce_loss = nn.BCEWithLogitsLoss()
    bce_loss = bce_loss(y_pred, y_true)
    return bce_loss
