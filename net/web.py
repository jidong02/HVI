"""
WEB: Wavelet Enhancement Block (WWE-UIE, Eq. 2-5, Lines 61-85 of model.py).

Haar wavelet decomposition → 1x1 fusion → SepConv refinement → upsample + residual.
Exact implementation from chingheng0808/WWE-UIE source.

Ref: Peng et al., "WWE-UIE: Wavelet-based Underwater Image Enhancement", 2025.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WaveletEnhanceBlock(nn.Module):
    """
    Haar wavelet-based multi-band enhancement block.

    Decomposes input features into 4 sub-bands (LL, LH, HL, HH),
    fuses them via 1x1 conv, refines with depthwise-separable conv,
    then upsamples and adds residual.

    Args:
        channels: number of input/output feature channels
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        # Haar wavelet kernels (4 filters, 2x2, per-channel groups)
        # Exact values from WWE-UIE source (model.py lines 65-68)
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        lh = torch.tensor([[-0.5, -0.5], [0.5, 0.5]])
        hl = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]])
        hh = torch.tensor([[0.5, -0.5], [-0.5, 0.5]])
        kernel = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)  # (4, 1, 2, 2)
        kernel = kernel.repeat(channels, 1, 1, 1)  # (4*C, 1, 2, 2)
        self.register_buffer("haar_kernel", kernel)

        # 1x1 conv: fuse 4*C → C (Eq. 4)
        self.fuse = nn.Conv2d(4 * channels, channels, kernel_size=1, bias=False)
        # Depthwise-separable conv for refinement (Eq. 5)
        self.post = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, 1, bias=False),
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) input feature map
        Returns:
            out: (B, C, H, W) wavelet-enhanced features
        """
        B, C, H, W = x.shape
        residual = x

        # 1. DWT: Haar wavelet decomposition with stride=2 (Eq. 3)
        dwt = F.conv2d(x, self.haar_kernel, stride=2, groups=C)  # (B, 4*C, H/2, W/2)

        # 2. Fuse: 1x1 conv (Eq. 4)
        fea = self.fuse(dwt)  # (B, C, H/2, W/2)

        # 3. Refine: depthwise-separable conv (Eq. 5, DWS-HINB simplified)
        fea = self.post(fea)  # (B, C, H/2, W/2)

        # 4. Upsample to original size (Eq. 5)
        out = F.interpolate(fea, size=(H, W), mode="bilinear", align_corners=False)

        return out + residual
