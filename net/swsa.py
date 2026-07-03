"""
SWSA (Simplified Window Spectral Self-Attention) for I-channel enhancement.

A lightweight FFT-based frequency gating module.
Inserts into the I-branch bottleneck of CIDNet.
Keeps HV (chroma polar-coordinate) branch untouched.

Architecture:
    FFT2D → Real/Imag split → Conv1x1 → IFFT2D → Residual add

This is a SIMPLIFIED version for quick verification.
SS-UIE's full SWSA uses learnable window masks; this version uses
a simple frequency-domain gate (conv1x1 on complex spectrum).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FreqGate(nn.Module):
    """
    Frequency-domain gating: learnable per-channel modulation in Fourier space.

    Input:  (B, C, H, W) feature map
    Output: (B, C, H, W) enhanced feature map (residual connection handled externally)
    """

    def __init__(self, channels, reduction=4):
        super(FreqGate, self).__init__()
        mid = max(channels // reduction, 8)
        # Two 1x1 convs on the complex spectrum (treat real + imag as 2*channels)
        self.conv1 = nn.Conv2d(channels * 2, mid * 2, 1, bias=False)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(mid * 2, channels * 2, 1, bias=False)

        # Learnable scaling factor (starts near 0 for identity-like init)
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) real-valued feature map
        Returns:
            out: (B, C, H, W) frequency-enhanced features
        """
        B, C, H, W = x.shape

        # FFT to frequency domain
        xf = torch.fft.rfft2(x, norm='ortho')  # (B, C, H, W//2+1) complex

        # Split real and imaginary
        xf_real = xf.real  # (B, C, H, W//2+1)
        xf_imag = xf.imag

        # Concatenate as 2*C channels
        xf_cat = torch.cat([xf_real, xf_imag], dim=1)  # (B, 2*C, H, W//2+1)

        # Frequency gating via 1x1 convs
        xf_out = self.conv2(self.act(self.conv1(xf_cat)))

        # Split back to real and imaginary
        xf_out_real, xf_out_imag = torch.chunk(xf_out, 2, dim=1)
        xf_out_complex = torch.complex(xf_out_real, xf_out_imag)

        # IFFT back to spatial domain
        out = torch.fft.irfft2(xf_out_complex, s=(H, W), norm='ortho')

        # Scale and return (residual connection handled by caller)
        return out * self.scale


class IBranchSWSA(nn.Module):
    """
    SWSA applied specifically to the I-branch bottleneck of CIDNet.

    Placement: after I_LCA3 (encoder bottleneck), before I_LCA4 (decoder).
    Channels: ch4 = 144 (CIDNet default bottleneck).
    """

    def __init__(self, channels=144):
        super(IBranchSWSA, self).__init__()
        self.freq_gate = FreqGate(channels)

    def forward(self, i_enc4, hv_4):
        """
        Args:
            i_enc4: (B, ch4, H/8, W/8) I-branch bottleneck features
            hv_4:   (B, ch4, H/8, W/8) HV-branch bottleneck (not modified, passed through)
        Returns:
            i_enc4: enhanced I-branch features (with frequency gating residual)
            hv_4:   unchanged HV-branch features
        """
        freq_residual = self.freq_gate(i_enc4)
        i_enc4 = i_enc4 + freq_residual
        return i_enc4, hv_4
