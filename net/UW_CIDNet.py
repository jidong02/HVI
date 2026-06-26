"""
UWCIDNet: CIDNet with UW-HVI pre/post transform.

Wraps the physics-grounded g/g⁻¹ around the existing HVIT/PHVIT pipeline.
The CIDNet architecture, HVIT/PHVIT math, and losses are UNCHANGED.

Flow:
    I (sRGB) → g → u (rebalanced) → HVIT → HVI
    → CIDNet encoder-decoder (unchanged)
    → PHVIT → u' (rebalanced sRGB) → g⁻¹ → Ĵ (final enhanced sRGB)
"""

import torch
import torch.nn as nn
from net.transformer_utils import *
from net.LCA import *
from net.lf_corrector import ILowFreqCorrector
from net.UW_HVI import UWHVITransform
from huggingface_hub import PyTorchModelHubMixin


class UWCIDNet(nn.Module, PyTorchModelHubMixin):
    """
    CIDNet with UW-HVI pre/post transform.

    Supports ablation modes via use_depth flag:
        - use_depth=True:  full UW-HVI with learned depth head (A3)
        - use_depth=False: global IFM rebalance, d(x) ≡ 1 (A1)
    """

    def __init__(self,
                 channels=[36, 36, 72, 144],
                 heads=[1, 2, 4, 8],
                 norm=False,
                 use_lfrc=False,
                 use_depth=True
                 ):
        super(UWCIDNet, self).__init__()

        [ch1, ch2, ch3, ch4] = channels
        [head1, head2, head3, head4] = heads

        # HV_ways (encoder for H,V channels — 2 channels input after HVIT)
        self.HVE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(3, ch1, 3, stride=1, padding=0, bias=False)
        )
        self.HVE_block1 = NormDownsample(ch1, ch2, use_norm=norm)
        self.HVE_block2 = NormDownsample(ch2, ch3, use_norm=norm)
        self.HVE_block3 = NormDownsample(ch3, ch4, use_norm=norm)

        self.HVD_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.HVD_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.HVD_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.HVD_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 2, 3, stride=1, padding=0, bias=False)
        )

        # I_ways (encoder for I channel — 1 channel)
        self.IE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(1, ch1, 3, stride=1, padding=0, bias=False),
        )
        self.IE_block1 = NormDownsample(ch1, ch2, use_norm=norm)
        self.IE_block2 = NormDownsample(ch2, ch3, use_norm=norm)
        self.IE_block3 = NormDownsample(ch3, ch4, use_norm=norm)

        self.ID_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.ID_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.ID_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.ID_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 1, 3, stride=1, padding=0, bias=False),
        )

        # LCA blocks (unchanged)
        self.HV_LCA1 = HV_LCA(ch2, head2)
        self.HV_LCA2 = HV_LCA(ch3, head3)
        self.HV_LCA3 = HV_LCA(ch4, head4)
        self.HV_LCA4 = HV_LCA(ch4, head4)
        self.HV_LCA5 = HV_LCA(ch3, head3)
        self.HV_LCA6 = HV_LCA(ch2, head2)

        self.I_LCA1 = I_LCA(ch2, head2)
        self.I_LCA2 = I_LCA(ch3, head3)
        self.I_LCA3 = I_LCA(ch4, head4)
        self.I_LCA4 = I_LCA(ch4, head4)
        self.I_LCA5 = I_LCA(ch3, head3)
        self.I_LCA6 = I_LCA(ch2, head2)

        # UW-HVI transform (replaces RGB_HVI)
        # NOTE: trans contains the original RGB_HVI as trans.hvi for HVIT/PHVIT
        self.trans = UWHVITransform(use_depth=use_depth)

        self.use_lfrc = use_lfrc
        if self.use_lfrc:
            self.lfrc = ILowFreqCorrector()

    def forward(self, x):
        """
        Forward pass with UW-HVI wrapping.

        Args:
            x: (B, 3, H, W) sRGB low-light input image ∈ [0,1]
        Returns:
            output_rgb: (B, 3, H, W) enhanced sRGB image ∈ [0,1]

        Pipeline:
            I → g → u (rebalanced) → HVIT → HVI → CIDNet → PHVIT → u'
            → g⁻¹ → Ĵ (final sRGB)
        """
        dtypes = x.dtype

        # Step 1: g-transform (sRGB → rebalanced), cache t_c
        u = self.trans.forward_g(x)  # (B, 3, H, W), clamped to [0,1]

        # Step 2: HVIT (rebalanced → HVI) — uses original HVI math
        hvi = self.trans.HVIT(u)  # (B, 3, H, W): [H, V, I]

        # Step 3: Split channels
        i = hvi[:, 2, :, :].unsqueeze(1).to(dtypes)   # I (intensity/value)

        # === CIDNet encoder-decoder (IDENTICAL to baseline) ===
        # low
        i_enc0 = self.IE_block0(i)
        i_enc1 = self.IE_block1(i_enc0)
        hv_0 = self.HVE_block0(hvi)
        hv_1 = self.HVE_block1(hv_0)
        i_jump0 = i_enc0
        hv_jump0 = hv_0

        i_enc2 = self.I_LCA1(i_enc1, hv_1)
        hv_2 = self.HV_LCA1(hv_1, i_enc1)
        v_jump1 = i_enc2
        hv_jump1 = hv_2
        i_enc2 = self.IE_block2(i_enc2)
        hv_2 = self.HVE_block2(hv_2)

        i_enc3 = self.I_LCA2(i_enc2, hv_2)
        hv_3 = self.HV_LCA2(hv_2, i_enc2)
        v_jump2 = i_enc3
        hv_jump2 = hv_3
        i_enc3 = self.IE_block3(i_enc2)
        hv_3 = self.HVE_block3(hv_2)

        i_enc4 = self.I_LCA3(i_enc3, hv_3)
        hv_4 = self.HV_LCA3(hv_3, i_enc3)

        i_dec4 = self.I_LCA4(i_enc4, hv_4)
        hv_4 = self.HV_LCA4(hv_4, i_enc4)

        hv_3 = self.HVD_block3(hv_4, hv_jump2)
        i_dec3 = self.ID_block3(i_dec4, v_jump2)
        i_dec2 = self.I_LCA5(i_dec3, hv_3)
        hv_2 = self.HV_LCA5(hv_3, i_dec3)

        hv_2 = self.HVD_block2(hv_2, hv_jump1)
        i_dec2 = self.ID_block2(i_dec2, v_jump1)

        i_dec1 = self.I_LCA6(i_dec2, hv_2)
        hv_1 = self.HV_LCA6(hv_2, i_dec2)

        i_dec1 = self.ID_block1(i_dec1, i_jump0)
        i_dec0 = self.ID_block0(i_dec1)
        hv_1 = self.HVD_block1(hv_1, hv_jump0)
        hv_0 = self.HVD_block0(hv_1)

        if self.use_lfrc:
            i_dec0, hv_0 = self.lfrc(i_dec0, hv_0, x)

        # Residual connection in HVI space
        output_hvi = torch.cat([hv_0, i_dec0], dim=1) + hvi

        # Step 4: PHVIT (HVI → rebalanced sRGB) — uses original HVI math
        u_prime = self.trans.PHVIT(output_hvi)  # (B, 3, H, W)

        # Step 5: g⁻¹ transform (rebalanced → sRGB) using cached t_c
        output_rgb = self.trans.forward_g_inv(u_prime)  # (B, 3, H, W)

        return output_rgb

    def HVIT(self, x):
        """
        Convenience method: sRGB → HVI.
        Uses the full g → HVIT chain. For loss computation compatibility.
        """
        u = self.trans.forward_g(x)
        return self.trans.HVIT(u)
