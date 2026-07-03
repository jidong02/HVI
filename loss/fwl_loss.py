"""
FWL: Frequency Weighted Loss (SS-UIE, Eq. 11-15).

Frequency-domain loss with dynamic weighting to emphasize
hard-to-reconstruct high frequencies.

Ref: Peng et al., "Adaptive Dual-domain Learning for UIE", AAAI 2025.
"""

import torch
import torch.nn as nn
import torch.fft


class FWLLoss(nn.Module):
    """
    Frequency Weighted Loss.

    For each channel: FFT → compute spectral distance → weight by |d|^0.5

    L_FWL = mean_{H,W} (θ · d)  summed over channels
    where d = |F_gt - F_pred|^2,  θ = sqrt(|d|).detach()
    """

    def __init__(self):
        super(FWLLoss, self).__init__()

    def forward(self, pred, gt):
        """
        Args:
            pred: (B, C, H, W) predicted image
            gt:   (B, C, H, W) ground truth image
        Returns:
            loss: scalar, mean over batch
        """
        # 1. FFT on H, W dimensions (last two dims)
        F_pred = torch.fft.fft2(pred, dim=(-2, -1))
        F_gt = torch.fft.fft2(gt, dim=(-2, -1))

        # 2. Spectral distance: |F_gt - F_pred|^2
        d_real = F_gt.real - F_pred.real
        d_imag = F_gt.imag - F_pred.imag
        d = d_real ** 2 + d_imag ** 2  # (B, C, H, W), power distance

        # 3. Dynamic weight: θ = sqrt(|d|), detached
        theta = torch.sqrt(torch.abs(d)).detach()

        # 4. Weighted mean over H,W, sum over C
        # L_FWL = sum_c [ mean_{H,W} (θ_c · d_c) ]
        loss_per_channel = (theta * d).mean(dim=(-2, -1))  # (B, C)
        loss = loss_per_channel.sum(dim=-1).mean()  # mean over batch

        return loss
