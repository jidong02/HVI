"""
DICAM modules ported from https://github.com/hfarhaditolie/DICAM
plus our DualPriorFusion (CAM-style gated fusion).

The DICAM class is kept structurally identical to the original
so that the pretrained UIEB checkpoint can be loaded as-is.
"""
import torch
import torch.nn as nn


# ============================================================
#  Original DICAM building blocks (do NOT modify, weight-compat)
# ============================================================
class Inc(nn.Module):
    def __init__(self, in_channels, filters):
        super(Inc, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, filters, (1, 1), (1, 1), dilation=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(filters, filters, (3, 3), (1, 1), dilation=1, padding=1),
            nn.LeakyReLU(),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, filters, (1, 1), (1, 1), dilation=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(filters, filters, (5, 5), (1, 1), dilation=1, padding=2),
            nn.LeakyReLU(),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.Conv2d(in_channels, filters, (1, 1), (1, 1), dilation=1),
            nn.LeakyReLU(),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, filters, (1, 1), (1, 1), dilation=1),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        o1 = self.branch1(x)
        o2 = self.branch2(x)
        o3 = self.branch3(x)
        o4 = self.branch4(x)
        return torch.cat([o1, o2, o3, o4], dim=1)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class CAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio):
        super(CAM, self).__init__()
        self.module = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.Softsign(),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Softsign(),
        )

    def forward(self, x):
        return x * self.module(x).unsqueeze(2).unsqueeze(3).expand_as(x)


class DICAM(nn.Module):
    """Full DICAM network, structurally identical to the official repo."""
    def __init__(self):
        super(DICAM, self).__init__()
        self.layer_1_r = Inc(in_channels=1, filters=64)
        self.layer_1_g = Inc(in_channels=1, filters=64)
        self.layer_1_b = Inc(in_channels=1, filters=64)

        self.layer_2_r = CAM(256, 4)
        self.layer_2_g = CAM(256, 4)
        self.layer_2_b = CAM(256, 4)

        self.layer_3 = Inc(768, 64)
        self.layer_4 = CAM(256, 4)

        self.layer_tail = nn.Sequential(
            nn.Conv2d(256, 24, (3, 3), (1, 1), padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(24, 3, (1, 1), (1, 1), padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        r = x[:, 0:1, :, :]
        g = x[:, 1:2, :, :]
        b = x[:, 2:3, :, :]

        r = self.layer_2_r(self.layer_1_r(r))
        g = self.layer_2_g(self.layer_1_g(g))
        b = self.layer_2_b(self.layer_1_b(b))

        feat = torch.cat([r, g, b], dim=1)
        feat = self.layer_4(self.layer_3(feat))
        return self.layer_tail(feat)


# ============================================================
#  Ours: CAM-style Dual-Prior Fusion
# ============================================================
class DualPriorFusion(nn.Module):
    """
    Adaptive fusion of DICAM output (RGB color-attenuation prior)
    and raw input (HVI-ready signal).

    Design:
      - Channel-gate: GAP -> Conv -> Softsign -> Conv -> Sigmoid (CAM-style)
      - Spatial-gate: Conv3x3 -> LeakyReLU -> Conv1x1 -> Sigmoid
      - Combined gate = ch_gate * sp_gate    (in [0, 1])
      - Output = gate * x_dicam + (1 - gate) * x_raw

    The use of Softsign as the inner activation mirrors DICAM's CAM
    block (paper sec. 2.4), making the design lineage explicit.
    """
    def __init__(self, in_channels=6, reduction=4, mid_channels=16):
        super().__init__()
        # CAM-style channel-wise gate
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, max(in_channels // reduction, 1), 1),
            nn.Softsign(),
            nn.Conv2d(max(in_channels // reduction, 1), 3, 1),
            nn.Sigmoid(),
        )
        # Spatial gate
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(mid_channels, 3, 1),
            nn.Sigmoid(),
        )

    def forward(self, x_dicam, x_raw):
        cat = torch.cat([x_dicam, x_raw], dim=1)  # B, 6, H, W
        ch_gate = self.channel_gate(cat)          # B, 3, 1, 1
        sp_gate = self.spatial_gate(cat)          # B, 3, H, W
        gate = ch_gate * sp_gate                  # B, 3, H, W
        fused = gate * x_dicam + (1.0 - gate) * x_raw
        return torch.clamp(fused, 0.0, 1.0)

    @torch.no_grad()
    def get_gates(self, x_dicam, x_raw):
        """Return (ch_gate, sp_gate, gate) for paper visualization."""
        cat = torch.cat([x_dicam, x_raw], dim=1)
        ch_gate = self.channel_gate(cat)
        sp_gate = self.spatial_gate(cat)
        return ch_gate, sp_gate, ch_gate * sp_gate
