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

        class CrossEntropy:
    """
    y_hat: [N, num_classes]

    y: [N]
    """

    name = "cross entropy"

    def __init__(
        self, num_classes: int, smoothing: float = 0.2, class_weights: Tensor = None
    ) -> None:
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.class_weights = class_weights

    def __call__(self, y_hat: Tensor, y: Tensor) -> Tensor:
        """
        Calculate cross entropy loss, apply label smoothing if needed.

        Parameters
        ----------
        y_hat : Tensor
            The predicted output tensor.
        y : Tensor
            The ground truth tensor.
        smoothing : bool
            Whether to apply label smoothing.

        Returns
        -------
        Tensor
            The loss tensor.
        """
        # bs = y.shape[0]
        y_hat_reshaped = y_hat.view(-1, self.num_classes)
        y_reshaped = y.view(-1, 1).contiguous().squeeze().view(-1)

        return F.cross_entropy(
            y_hat_reshaped,
            y_reshaped.long(),
            reduction="mean",
            weight=self.class_weights
        )

    def __eq__(self, other):
        return self.name == other
