import torch
import torch.nn as nn
from einops import rearrange
from net.transformer_utils import *


# ============== NEW: Channel-aware Channel Attention ==============
class CCA(nn.Module):
    """
    使用 GAP + GMP + Var 三种通道统计的注意力。
    比标准 SE 多了通道方差，对水下场景的通道不平衡更敏感。
    """
    def __init__(self, dim, reduction=4):
        super().__init__()
        hidden = max(dim // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Linear(dim * 3, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, dim),
        )
        self.sigmoid = nn.Sigmoid()
        
        # 初始化为零，sigmoid(0)=0.5，让 CCA 初始等价于 *0.5（仍然是恒等的常数缩放）
        # 实际我们想初始 attn ≈ 1（不影响预训练），所以把最后 layer 的 bias 设大正数
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.constant_(self.mlp[-1].bias, 4.0)  # sigmoid(4) ≈ 0.98
    
    def forward(self, x):
        b, c, _, _ = x.shape
        avg = x.mean(dim=(2, 3))                    # (B, C)
        mx = x.amax(dim=(2, 3))                     # (B, C)
        var = x.var(dim=(2, 3), unbiased=False)     # (B, C)
        stats = torch.cat([avg, mx, var], dim=1)    # (B, 3C)
        attn = self.sigmoid(self.mlp(stats)).view(b, c, 1, 1)
        return x * attn

class CAB(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(CAB, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):
        b, c, h, w = x.shape
        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = nn.functional.softmax(attn, dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class IEL(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False):
        super(IEL, self).__init__()
        hidden_features = int(dim*ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)
        self.dwconv1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
        self.dwconv2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        self.Tanh = nn.Tanh()
    
    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x1 = self.Tanh(self.dwconv1(x1)) + x1
        x2 = self.Tanh(self.dwconv2(x2)) + x2
        x = x1 * x2
        x = self.project_out(x)
        return x


# ============== HV_LCA：加入 CCA ==============
class HV_LCA(nn.Module):
    def __init__(self, dim, num_heads, bias=False, use_cca=True):
        super(HV_LCA, self).__init__()
        self.gdfn = IEL(dim)
        self.norm = LayerNorm(dim)
        self.ffn = CAB(dim, num_heads, bias)
        self.use_cca = use_cca
        if use_cca:
            self.cca = CCA(dim)
        
    def forward(self, x, y):
        x = x + self.ffn(self.norm(x), self.norm(y))
        x = self.gdfn(self.norm(x))
        if self.use_cca:
            x = self.cca(x)
        return x


# ============== I_LCA：加入 CCA ==============
class I_LCA(nn.Module):
    def __init__(self, dim, num_heads, bias=False, use_cca=True):
        super(I_LCA, self).__init__()
        self.norm = LayerNorm(dim)
        self.gdfn = IEL(dim)
        self.ffn = CAB(dim, num_heads, bias=bias)
        self.use_cca = use_cca
        if use_cca:
            self.cca = CCA(dim)
        
    def forward(self, x, y):
        x = x + self.ffn(self.norm(x), self.norm(y))
        x = x + self.gdfn(self.norm(x))
        if self.use_cca:
            x = self.cca(x)
        return x