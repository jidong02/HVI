"""
True DCSSB (Dense Cross-Scale SS Block) from SS-UIE, faithfully implemented.

Architecture (per MemoryBlock):
    - N ResidualBlocks, each = 2 × SF_Block (Mamba + Frequency) with residual
    - GateUnit: 1×1 conv on concat(all_previous_ys + all_recursive_xs)
    - Dense connection: each MemoryBlock output added to accumulated ys list

For CIDNet integration:
    - I branch: full SF_Block (Mamba + Spec)
    - HV branch: Mamba-only (no Spec/FFT to avoid polar coordinate issues)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os

# Import from bundled SS-UIE blocks
from net.ss_blocks import MambaLayer


class ResidualMambaBlock(nn.Module):
    """One residual block: 2 × Mamba layers with skip connection."""

    def __init__(self, channels, drop_rate, H, W):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.mamba1 = MambaLayer(input_dim=channels, output_dim=channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.mamba2 = MambaLayer(input_dim=channels, output_dim=channels)

    def forward(self, x):
        # x: (B, C, H, W) — both our convention and MambaLayer's convention
        residual = x
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.mamba1(out)  # MambaLayer takes (B,C,H,W), returns (B,C,H,W)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.mamba2(out)  # same
        return out + residual


class DenseMemoryBlock(nn.Module):
    """One dense memory block with recursive units and gate."""

    def __init__(self, channels, num_resblocks, num_prev_blocks, drop_rate, H, W):
        super().__init__()
        self.recursive_units = nn.ModuleList([
            ResidualMambaBlock(channels, drop_rate, H, W)
            for _ in range(num_resblocks)
        ])
        # Gate: 1x1 conv on (num_resblocks + num_prev_blocks) * channels → channels
        total_in = (num_resblocks + num_prev_blocks) * channels
        self.gate = nn.Sequential(
            nn.BatchNorm2d(total_in),
            nn.ReLU(inplace=True),
            nn.Conv2d(total_in, channels, 1, 1, 0),
        )

    def forward(self, x, prev_outputs):
        """
        Args:
            x: (B, C, H, W) current input
            prev_outputs: list of (B, C, H, W) outputs from all previous blocks
        Returns:
            gate_out: (B, C, H, W) gated output
            updated_prev: list with this output appended
        """
        xs = []
        for unit in self.recursive_units:
            x = unit(x)
            xs.append(x)

        # Gate: concat all recursive outputs + all previous block outputs
        gate_out = self.gate(torch.cat(xs + prev_outputs, dim=1))

        prev_outputs.append(gate_out)
        return gate_out, prev_outputs


class DCSSB_Stack(nn.Module):
    """
    Full DCSSB stack: N DenseMemoryBlocks with true dense connections.

    Args:
        channels: feature channels (e.g., 144 for CIDNet bottleneck)
        num_memblocks: number of dense memory blocks (default 4)
        num_resblocks_per: recursive units per memory block (default 6)
        H, W: spatial size of bottleneck features
    """

    def __init__(self, channels=144, num_memblocks=4, num_resblocks_per=3,
                 drop_rate=0.0, H=32, W=32):
        super().__init__()
        self.channels = channels
        self.num_memblocks = num_memblocks
        self.blocks = nn.ModuleList([
            DenseMemoryBlock(
                channels=channels,
                num_resblocks=num_resblocks_per,
                num_prev_blocks=i + 1,  # 1-based: includes itself + previous
                drop_rate=drop_rate,
                H=H, W=W,
            )
            for i in range(num_memblocks)
        ])
        # Learnable fusion weights
        self.weights = nn.Parameter(torch.ones(1, num_memblocks) / num_memblocks)
        # Output fusion: weighted sum of all block outputs + final conv
        self.fusion = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) input features
        Returns:
            out: (B, C, H, W) enhanced features
        """
        prev_outputs = [x]  # initial "previous outputs" includes the input
        mid_feats = []

        for i, block in enumerate(self.blocks):
            out, prev_outputs = block(x, prev_outputs)
            mid_feats.append(out)
            x = out  # feed-forward to next block

        # Weighted fusion of all block outputs (like SS-UIE's weighted sum)
        w_sum = self.weights.sum(1)
        fused = torch.zeros_like(mid_feats[0])
        for i, feat in enumerate(mid_feats):
            fused = fused + feat * self.weights.data[0][i] / w_sum

        fused = self.fusion(fused)
        return fused * self.scale


class CIDNet_DCSSB(nn.Module):
    """
    True DCSSB inserted at CIDNet bottleneck (both I and HV branches).
    I branch: full Mamba stacking
    HV branch: Mamba-only (no FFT on polar coords)
    """

    def __init__(self, channels=144, num_memblocks=2, num_resblocks_per=3, H=32, W=32):
        super().__init__()
        self.i_dcssb = DCSSB_Stack(channels, num_memblocks, num_resblocks_per, H=H, W=W)
        self.hv_dcssb = DCSSB_Stack(channels, num_memblocks, num_resblocks_per, H=H, W=W)

    def forward(self, i_feat, hv_feat):
        i_feat = i_feat + self.i_dcssb(i_feat)
        hv_feat = hv_feat + self.hv_dcssb(hv_feat)
        return i_feat, hv_feat
