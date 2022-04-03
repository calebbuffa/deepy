from __future__ import annotations

import torch.nn as nn
import torch
from torch import Tensor
import torch.nn.functional as F

from metrics import multiclass_dice_coeff, dice_coeff


class DiceLoss:
    def __init__(self, n_classes: int = 2):
        self.n_classes = n_classes
        self.multiclass = self.n_classes > 1

    def __call__(self, y_hat: Tensor, y: Tensor) -> Tensor:
        y_hat = F.softmax(y_hat, dim=1)
        y = F.one_hot(y.long(), self.n_classes).permute(0, 3, 1, 2).float()
        if self.multiclass:
            return 1 - multiclass_dice_coeff(y_hat, y, reduce_batch_first=True)
        else:
            return 1 - dice_coeff(y_hat, y, reduce_batch_first=True)

class FocalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        ...
    
    def forward(self, y_hat: Tensor, y: Tensor) -> Tensor:
        ...
