"""
L_HVI: Charbonnier loss in HVI color space (WWE-UIE, Eq. 15).

Computes L1-Charbonnier between HVIT(pred) and HVIT(gt).
The HVIT function converts RGB → (H, V, I) perceptual color space.

Ref: Yan et al., "HVI: A New color space for Low-light Image Enhancement", CVPR 2025.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HVILoss(nn.Module):
    """
    Charbonnier loss in HVI space.

    L_HVI = mean( sqrt((HVI_gt - HVI_pred)^2 + eps^2) )

    Args:
        hv_transform: RGB_HVI module or function that maps RGB → HVI
        eps: smoothing constant for Charbonnier (default 1e-3)
    """

    def __init__(self, hv_transform, eps=1e-3):
        super(HVILoss, self).__init__()
        self.hv_transform = hv_transform
        self.eps = eps

    def forward(self, pred, gt):
        """
        Args:
            pred: (B, 3, H, W) predicted RGB image in [0, 1]
            gt:   (B, 3, H, W) ground truth RGB image in [0, 1]
        Returns:
            loss: scalar
        """
        hvi_pred = self.hv_transform.HVIT(pred)
        hvi_gt = self.hv_transform.HVIT(gt)
        diff = hvi_gt - hvi_pred
        loss = torch.sqrt(diff ** 2 + self.eps ** 2).mean()
        return loss
