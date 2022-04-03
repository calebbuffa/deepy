from __future__ import annotations
import torch.nn as nn
import torch
from torch import Tensor
import torch.nn.functional as F

def dice_coeff(
    y_hat: Tensor, y: Tensor, reduce_batch_first: bool = False, epsilon=1e-6
):
    # Average of Dice coefficient for all batches, or for a single mask
    assert y_hat.size() == y.size(), print(y_hat.size(), y.size())
    if y_hat.dim() == 2 and reduce_batch_first:
        raise ValueError(
            f"Dice: asked to reduce batch but got tensor without batch dimension (shape {y_hat.shape})"
        )

    if y_hat.dim() == 2 or reduce_batch_first:
        inter = torch.dot(y_hat.reshape(-1), y.reshape(-1))
        sets_sum = torch.sum(y_hat) + torch.sum(y)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = sum(dice_coeff(y_hat[i, ...], y[i, ...]) for i in range(y_hat.shape[0]))

        return dice / y_hat.shape[0]


def multiclass_dice_coeff(
    y_hat: Tensor, y: Tensor, reduce_batch_first: bool = False, epsilon=1e-6
):
    # Average of Dice coefficient for all classes
    assert y_hat.size() == y.size()
    dice = sum(
        dice_coeff(
            y_hat[:, channel, ...], y[:, channel, ...], reduce_batch_first, epsilon,
        )
        for channel in range(y_hat.shape[1])
    )

    return dice / y_hat.shape[1]


class DiceLoss:
    def __init__(self, n_classes: int = 2):
        self.n_classes = n_classes
        self.multiclass = self.n_classes > 1

    def __call__(self, y_hat: Tensor, y: Tensor) -> Tensor:
        y_hat = F.softmax(y_hat, dim=1)
        y = F.one_hot(y.long(), self.n_classes).permute(0, 3, 1, 2).float()
        if self.multiclass:
            return multiclass_dice_coeff(y_hat, y, reduce_batch_first=True)
        else:
            return dice_coeff(y_hat, y, reduce_batch_first=True)

class FocalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        ...
    
    def forward(self, y_hat: Tensor, y: Tensor) -> Tensor:
        ...
