import torch
import torch.nn.functional as F
from torch import nn

class BCELoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, y_pred, y_true):
        return self.bce(y_pred.contiguous().view(-1), y_true.contiguous().view(-1))

class DiceLoss(nn.Module):

    def forward(self, y_pred, y_true):
        y_pred = y_pred.contiguous().view(-1)
        y_true = y_true.contiguous().view(-1)
        inter = (y_true * y_pred).sum()
        return 1 - (2 * inter + 1.0) / (y_pred.sum() + y_true.sum() + 1.0)

class FocalTverskyLoss(nn.Module):

    def __init__(self, alpha: float=0.6, beta: float=0.4, gamma: float=2.5, smooth: float=1e-05):
        super().__init__()
        self.alpha, self.beta, self.gamma, self.smooth = (alpha, beta, gamma, smooth)

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)
        if targets.dim() == 3:
            targets = targets.unsqueeze(1)
        b = preds.size(0)
        preds_flat = preds.reshape(b, -1)
        targets_flat = targets.reshape(b, -1)
        tp = (preds_flat * targets_flat).sum(dim=1)
        fp = (preds_flat * (1 - targets_flat)).sum(dim=1)
        fn = ((1 - preds_flat) * targets_flat).sum(dim=1)
        tversky = tp / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return ((1 - tversky) ** self.gamma).mean()

class BceDiceLoss(nn.Module):

    def __init__(self, wb=1.0, wd=1.0):
        super().__init__()
        self.wb, self.wd = (float(wb), float(wd))
        self.bce, self.dice = (BCELoss(), DiceLoss())

    def forward(self, y_pred, y_true):
        return self.wd * self.dice(y_pred, y_true) + self.wb * self.bce(y_pred, y_true)

class EccoDiceLoss(nn.Module):

    def forward(self, pred, target):
        smooth = 1
        size = pred.size(0)
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + smooth) / (pred_.sum(1) + target_.sum(1) + smooth)
        return 1 - dice_score.sum() / size

class EccoBceDiceLoss(nn.Module):

    def __init__(self, wb=1.0, wd=1.0):
        super().__init__()
        self.bce = BCELoss()
        self.dice = EccoDiceLoss()
        self.wb, self.wd = (float(wb), float(wd))

    def forward(self, pred, target):
        return self.wd * self.dice(pred, target) + self.wb * self.bce(pred, target)

class UNetDiceLoss(nn.Module):

    def forward(self, inputs, target):
        inputs = torch.sigmoid(inputs)
        if target.dim() == 3:
            target = target.unsqueeze(1)
        p = inputs.view(inputs.size(0), -1)
        t = target.view(target.size(0), -1)
        inter = (p * t).sum(dim=1)
        union = p.sum(dim=1) + t.sum(dim=1)
        return (1.0 - (2.0 * inter + 1e-05) / (union + 1e-05)).mean()

class UNetBceDiceLoss(nn.Module):

    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = UNetDiceLoss()
        self.bce_weight, self.dice_weight = (float(bce_weight), float(dice_weight))

    def forward(self, pred, target):
        if target.dim() == 3:
            target = target.unsqueeze(1)
        return self.bce_weight * self.bce(pred, target.float()) + self.dice_weight * self.dice(pred, target.float())

class MultiClassDiceLoss(nn.Module):

    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = int(n_classes)

    def _one_hot_encoder(self, input_tensor):
        return torch.cat([(input_tensor == i).unsqueeze(1) for i in range(self.n_classes)], dim=1).float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-05
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        return 1 - (2 * intersect + smooth) / (z_sum + y_sum + smooth)

    def forward(self, inputs, target):
        inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        loss = 0.0
        for i in range(self.n_classes):
            loss += self._dice_loss(inputs[:, i], target[:, i])
        return loss / self.n_classes

class CEDiceLoss(nn.Module):

    def __init__(self, n_classes=2, ce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = MultiClassDiceLoss(n_classes)
        self.ce_weight, self.dice_weight = (float(ce_weight), float(dice_weight))

    def forward(self, logits, target):
        return self.ce_weight * self.ce(logits, target) + self.dice_weight * self.dice(logits, target)

class SoftDiceLoss(nn.Module):

    def __init__(self, smooth: float=0.01):
        super().__init__()
        self.smooth = float(smooth)

    def forward(self, pred, target):
        nc = pred.shape[1]
        prob = F.softmax(pred, dim=1).reshape(pred.shape[0], nc, -1)
        oh = F.one_hot(target.long(), num_classes=nc).permute(0, 3, 1, 2)
        oh = oh.reshape(pred.shape[0], nc, -1).float()
        inter = (prob * oh).sum(2) + self.smooth
        union = prob.sum(2) + oh.sum(2) + self.smooth
        return 1.0 - (2.0 * inter / union).sum() / (pred.shape[0] * nc)
LOSSES = {'bce_dice': BceDiceLoss, 'ecco_bce_dice': EccoBceDiceLoss, 'soft_dice': SoftDiceLoss, 'unet_bce_dice': UNetBceDiceLoss, 'focal_tversky': FocalTverskyLoss, 'ce_dice': CEDiceLoss}

def build_loss(name, **kw):
    if name not in LOSSES:
        raise KeyError(f'未知 loss: {name};可用 {list(LOSSES)}')
    return LOSSES[name](**kw)
