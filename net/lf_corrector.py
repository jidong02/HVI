"""LFRC v3 — spatial I-only corrector at 1/8 resolution. ~5K params, identity-init, HV untouched."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv(cin, cout, k=3, s=1, act=True):
    layers = [nn.Conv2d(cin, cout, k, s, k // 2)]
    if act:
        layers += [nn.GELU()]
    return nn.Sequential(*layers)


class ILowFreqCorrector(nn.Module):
    """Spatial intensity corrector at 1/8 resolution.
    Identity at init: last conv zero-init → gain=0, bias=0 → i_out = i."""
    def __init__(self, base=16):
        super().__init__()
        cin = 4  # RGB(3) + I(1)
        self.enc = nn.Sequential(
            _conv(cin, base, s=2),  # 1/2
            _conv(base, base, s=2),  # 1/4
            _conv(base, base, s=2),  # 1/8
        )
        self.proc = nn.Sequential(_conv(base, base), _conv(base, base, act=False))
        self.head_gain = nn.Conv2d(base, 1, 3, 1, 1)
        self.head_bias = nn.Conv2d(base, 1, 3, 1, 1)
        nn.init.zeros_(self.head_gain.weight); nn.init.zeros_(self.head_gain.bias)
        nn.init.zeros_(self.head_bias.weight); nn.init.zeros_(self.head_bias.bias)
        self.scale = 0.05

    def forward(self, i, hv, x):
        B, _, H, W = i.shape
        f = torch.cat([x, i], dim=1)         # (B, 4, H, W)
        f = self.enc(f)                       # (B, base, H/8, W/8)
        f = self.proc(f)
        g = torch.tanh(self.head_gain(f)) * self.scale  # bounded [-s, s]
        b = torch.tanh(self.head_bias(f)) * self.scale
        g = F.interpolate(g, size=(H, W), mode='bilinear', align_corners=False)
        b = F.interpolate(b, size=(H, W), mode='bilinear', align_corners=False)
        i_out = i * (1.0 + g) + b
        return i_out, hv  # hv unchanged


# ----- self-test -----
if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, W = 2, 64, 64
    i = torch.rand(B, 1, H, W)
    hv = torch.rand(B, 2, H, W)
    x = torch.rand(B, 3, H, W)
    net = ILowFreqCorrector()

    # 1) identity
    i_o, hv_o = net(i, hv, x)
    id_err = (i_o - i).abs().max().item()
    print(f"[identity] max |i_out - i| = {id_err:.3e}  (should be 0)")

    # 2) gradient
    net2 = ILowFreqCorrector()
    i2, _ = net2(i, hv, x)
    loss = i2.mean()
    loss.backward()
    gnorm = net2.head_bias.weight.grad.abs().sum().item()
    print(f"[alive] last layer grad = {gnorm:.3e}  (>0)")

    # 3) params
    n = sum(p.numel() for p in net.parameters())
    print(f"[params] {n}")

    # 4) hv not touched
    print(f"[hv-safe] max |hv_out - hv| = {(hv_o - hv).abs().max().item():.3e}  (should be 0)")
    print("OK" if id_err < 1e-6 and gnorm > 0 else "FAIL")
