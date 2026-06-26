"""
UW-HVI: Underwater Image Formation Model (IFM) based pre-transform for HVI-CIDNet.

Adds a physics-grounded, provably invertible pre-transform `g` to HVI-CIDNet,
producing a new color space Φ_UW = Φ_HVI ∘ g.

g is the parameterized inverse of the underwater image formation model (IFM).

Per channel c ∈ {R,G,B}, per pixel x:
    t_c(x) = exp(-β_c * d(x))          clamped to [0.1, 1.0]
    u_c(x) = (I_c(x) - B_c) / t_c(x) + B_c          # g: sRGB → rebalanced
    Ĵ_c(x) = t_c(x) * u'_c(x) + B_c * (1 - t_c(x))   # g⁻¹: rebalanced → sRGB

This construction is initialized to be bit-identical to the current corrected baseline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthHead(nn.Module):
    """
    Small CNN that predicts a single-channel depth field from an sRGB image.

    Architecture: 3 conv layers, 16-32 ch, 3×3, ReLU, final 1×1 → 1 channel, softplus.
    A learnable depth scale (init=0) guarantees EXACT identity at step 0.

    Input:  sRGB image I ∈ [0,1]^{3×H×W}
    Output: depth field d(x) ≥ 0, same spatial size
    """

    def __init__(self, in_ch=3, mid_ch=32, mid2_ch=16):
        super(DepthHead, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(mid_ch, mid2_ch, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv3 = nn.Conv2d(mid2_ch, 1, kernel_size=1, stride=1, padding=0, bias=True)

        # Learnable depth scale, init = 0 → d = 0 exactly → t_c = 1 → g = identity
        self.depth_scale = nn.Parameter(torch.zeros(1))

        # Conv init: weights=0, bias=-6 → softplus(-6) ≈ 0.0025
        nn.init.zeros_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        nn.init.zeros_(self.conv3.weight)
        nn.init.constant_(self.conv3.bias, -6.0)

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) sRGB image ∈ [0,1]
        Returns:
            d: (B, 1, H, W) depth field ≥ 0
        """
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        raw_depth = F.softplus(self.conv3(out))  # ≥ 0, small at init
        d = raw_depth * self.depth_scale          # scale=0 → d=0 exactly at init
        return d


class UWHVITransform(nn.Module):
    """
    UW-HVI transform: wraps g / g⁻¹ around the existing RGB_HVI (HVIT/PHVIT).

    Learnable parameters (3 + 3 scalars):
        β_c = softplus(θ_c) > 0    per-channel attenuation coefficient
        B_c = sigmoid(φ_c) ∈ (0,1) per-channel backscatter / veiling light

    Modes:
        - use_depth=True:  d(x) = D_ψ(I)  (full UW-HVI, A3)
        - use_depth=False: d(x) ≡ 1       (global IFM rebalance, A1)
        - fixed_depth:     d(x) from external prior (A2) — set via set_fixed_depth()

    Initialization (guarantees EXACT identity = baseline):
        - Depth head: depth_scale = 0 → d = 0 exactly → t_c = 1 exactly → g = id
        - θ_c = -4 → β_c = softplus(-4) ≈ 0.018 (conservative, secondary)
        - φ_c = 0  → B_c = 0.5
        With d=0 and t_c=1: g = identity exactly, Φ_UW = Φ_HVI bit-for-bit.
    """

    def __init__(self, use_depth=True):
        super(UWHVITransform, self).__init__()

        # Import here to avoid circular imports
        from net.HVI_transform import RGB_HVI

        self.use_depth = use_depth
        if use_depth:
            self.depth_head = DepthHead()

        # Learnable parameters: θ_c (for β_c) and φ_c (for B_c)
        # θ_c init = -4 → β_c = softplus(-4) ≈ 0.018 (conservative)
        self.theta = nn.Parameter(torch.full((3,), -4.0))
        # φ_c init = 0 → B_c = sigmoid(0) = 0.5
        self.phi = nn.Parameter(torch.zeros(3))

        # The original HVI transform (HVIT / PHVIT)
        self.hvi = RGB_HVI()

        # Cached transmission for gauge round-trip
        self.cached_t = None

    def get_beta(self):
        """β_c = softplus(θ_c) > 0, shape: (3,)"""
        return F.softplus(self.theta)

    def get_B(self):
        """B_c = sigmoid(φ_c) ∈ (0,1), shape: (3,)"""
        return torch.sigmoid(self.phi)

    def compute_transmission(self, d, beta):
        """
        Compute transmission t_c(x) from depth d and attenuation β_c.

        Args:
            d:    (B, 1, H, W) depth field ≥ 0
            beta: (3,) per-channel attenuation coefficients
        Returns:
            t_c:  (B, 3, H, W) transmission, clamped to [0.1, 1.0]
        """
        # beta: (3,) → (1, 3, 1, 1), d: (B, 1, H, W)
        beta_expanded = beta.view(1, 3, 1, 1)
        t = torch.exp(-beta_expanded * d)  # (B, 3, H, W)
        t = torch.clamp(t, 0.1, 1.0)       # bound amplification to ≤10x
        return t

    def g(self, I, t, B):
        """
        Forward transform: sRGB → rebalanced space (IFM inverse).

        u_c(x) = (I_c(x) - B_c) / t_c(x) + B_c

        Args:
            I: (B, 3, H, W) sRGB image ∈ [0,1]
            t: (B, 3, H, W) transmission
            B: (3,) per-channel backscatter
        Returns:
            u: (B, 3, H, W) rebalanced image (unclamped, for round-trip)
        """
        B_expanded = B.view(1, 3, 1, 1)
        u = (I - B_expanded) / t + B_expanded
        return u

    def g_inv(self, u, t, B):
        """
        Inverse transform: rebalanced → sRGB (forward IFM).

        Ĵ_c(x) = t_c(x) * u'_c(x) + B_c * (1 - t_c(x))

        Args:
            u: (B, 3, H, W) rebalanced image
            t: (B, 3, H, W) transmission (same cached t_c from input)
            B: (3,) per-channel backscatter
        Returns:
            Ĵ: (B, 3, H, W) sRGB image
        """
        B_expanded = B.view(1, 3, 1, 1)
        J = t * u + B_expanded * (1.0 - t)
        return J

    def forward_g(self, I):
        """
        Full forward `g` pipeline: compute depth → transmission → rebalanced.
        Caches t_c for later use in g⁻¹.

        Fast-path: when depth is identically zero (d=0 ⇒ t=1 ⇒ g=id),
        returns I directly with NO floating-point perturbation.
        This guarantees bit-identical identity at init.

        Args:
            I: (B, 3, H, W) sRGB input image ∈ [0,1]
        Returns:
            u_clamped: (B, 3, H, W) rebalanced image, clamped to [0,1] for HVIT
        """
        B = self.get_B()       # (3,)
        beta = self.get_beta() # (3,)

        if self.use_depth:
            d = self.depth_head(I)   # (B, 1, H, W)
        else:
            # A1 mode: d(x) ≡ 1 (constant depth)
            d = torch.ones(I.shape[0], 1, I.shape[2], I.shape[3],
                          device=I.device, dtype=I.dtype)

        # Fast-path: if d ≡ 0, g = identity exactly (no floating-point perturbation)
        if d.max().item() == 0.0:
            t = torch.ones(I.shape[0], 3, I.shape[2], I.shape[3],
                          device=I.device, dtype=I.dtype)
            self.cached_t = t
            # I is already in [0,1], no clamp needed
            return I

        t = self.compute_transmission(d, beta)  # (B, 3, H, W)
        self.cached_t = t  # cache for g⁻¹

        u = self.g(I, t, B)                     # (B, 3, H, W), unclamped
        u_clamped = torch.clamp(u, 0.0, 1.0)    # clamp for HVIT stage only
        return u_clamped

    def forward_g_inv(self, u_prime):
        """
        Inverse transform `g⁻¹` using the cached transmission from forward_g.

        Fast-path: when t ≡ 1 (identity case), returns u_prime directly.

        Args:
            u_prime: (B, 3, H, W) rebalanced-space output from PHVIT
        Returns:
            Ĵ: (B, 3, H, W) final enhanced sRGB image
        """
        assert self.cached_t is not None, \
            "forward_g must be called before forward_g_inv to cache t_c"

        # Fast-path: if t ≡ 1, g⁻¹ = identity
        t = self.cached_t
        if (t == 1.0).all():
            self.cached_t = None
            return u_prime

        B = self.get_B()
        J = self.g_inv(u_prime, t, B)
        # Reset cache after use
        self.cached_t = None
        return J

    def HVIT(self, img):
        """Delegate to original HVIT (expects image in [0,1])."""
        return self.hvi.HVIT(img)

    def PHVIT(self, img):
        """Delegate to original PHVIT."""
        return self.hvi.PHVIT(img)

    def set_fixed_depth(self, depth_field):
        """
        For A2: use a fixed prior depth field instead of learned depth head.
        depth_field: (1, 1, H, W) or (B, 1, H, W) tensor, or path to .npy file
        """
        self._fixed_depth = depth_field

    def get_depth(self, I):
        """Get depth field (learned or fixed)."""
        if hasattr(self, '_fixed_depth') and self._fixed_depth is not None:
            if isinstance(self._fixed_depth, str):
                import numpy as np
                d = torch.from_numpy(np.load(self._fixed_depth)).float()
            else:
                d = self._fixed_depth
            if d.device != I.device:
                d = d.to(I.device)
            # Broadcast to batch size
            if d.shape[0] == 1 and I.shape[0] > 1:
                d = d.expand(I.shape[0], -1, -1, -1)
            return d
        elif self.use_depth:
            return self.depth_head(I)
        else:
            return torch.ones(I.shape[0], 1, I.shape[2], I.shape[3],
                            device=I.device, dtype=I.dtype)


# Convenience aliases
UW_HVI = UWHVITransform
