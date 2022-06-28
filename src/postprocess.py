from __future__ import annotations

from torch import nn
from torch import Tensor

class MatrixNMS(nn.Module):
    def __init__(self, kernel='gaussian', sigma=2.0):
        """
        Matrix NMS for multi-class masks.
                
        Parameters
        ----------
        kernel : str
            'linear' or 'gauss' 
        sigma : float
            std in gaussian method
        """
        super().__init__()
        self.kernel = kernel
        self.sigma = sigma

    def forward(self, seg_masks, cate_labels, cate_scores, sum_masks=None) -> Tensor:
        """
        Parameters
        ----------
        seg_masks : Tensor
            shape (n, h, w)
        cate_labels : Tensor
            shape (n), mask labels in descending order
        cate_scores : Tensor 
            shape (n), mask scores in descending order
        sum_masks : Tensor
            The sum of seg_masks
            
        Returns
        -------
            Tensor
                cate_scores_update, tensors of shape (n)
        """
        n_samples = len(cate_labels)
        if n_samples == 0:
            return []
        if sum_masks is None:
            sum_masks = seg_masks.sum((1, 2)).float()
        seg_masks = seg_masks.reshape(n_samples, -1).float()
        # inter.
        inter_matrix = torch.mm(seg_masks, seg_masks.transpose(1, 0))
        # union.
        sum_masks_x = sum_masks.expand(n_samples, n_samples)
        # iou.
        iou_matrix = (inter_matrix / (sum_masks_x + sum_masks_x.transpose(1, 0) - inter_matrix)).triu(diagonal=1)
        # label_specific matrix.
        cate_labels_x = cate_labels.expand(n_samples, n_samples)
        label_matrix = (cate_labels_x == cate_labels_x.transpose(1, 0)).float().triu(diagonal=1)

        # IoU compensation
        compensate_iou, _ = (iou_matrix * label_matrix).max(0)
        compensate_iou = compensate_iou.expand(n_samples, n_samples).transpose(1, 0)

        # IoU decay 
        decay_iou = iou_matrix * label_matrix

        # matrix nms
        if self.kernel == 'gaussian':
            decay_matrix = torch.exp(-1 * self.sigma * (decay_iou ** 2))
            compensate_matrix = torch.exp(-1 * self.sigma * (compensate_iou ** 2))
            decay_coefficient, _ = (decay_matrix / compensate_matrix).min(0)
        elif self.kernel == 'linear':
            decay_matrix = (1-decay_iou)/(1-compensate_iou)
            decay_coefficient, _ = decay_matrix.min(0)
        else:
            raise NotImplementedError

        return cate_scores * decay_coefficient
