"""
验证 aggressive v1 checkpoint 的加载和前向传播.

测试:
  - 256x256 和 1024x1024 两种输入
  - 显存峰值
  - NaN/Inf 检查
  - 参数量分解
"""
import torch
from net.CIDNet import CIDNet

CHECKPOINT = 'weights/aggressive_v1/init.pth'

print("=" * 70)
print("验证 aggressive v1 checkpoint")
print("=" * 70)

# 1. 加载模型
print(f"\n[1/4] 加载 checkpoint: {CHECKPOINT}")
model = CIDNet().cuda()
state = torch.load(CHECKPOINT, map_location='cpu')
missing, unexpected = model.load_state_dict(state, strict=True)
print(f"  Missing: {len(missing)} (应该为 0)")
print(f"  Unexpected: {len(unexpected)} (应该为 0)")

if missing or unexpected:
    print("  ⚠️ 加载有问题!")
    for k in missing[:3]: print(f"    Missing: {k}")
    for k in unexpected[:3]: print(f"    Unexpected: {k}")
    exit(1)

# 2. 参数量分解
print("\n[2/4] 参数量分解:")
total_params = sum(p.numel() for p in model.parameters())
hvi_params = sum(p.numel() for n, p in model.named_parameters() if 'dicam' not in n)
dicam_feat_params = sum(p.numel() for n, p in model.named_parameters() if 'dicam_feat' in n)
dicam_proj_params = sum(p.numel() for n, p in model.named_parameters() if 'dicam_proj' in n)

print(f"  总参数量: {total_params:,}")
print(f"  HVI 主干: {hvi_params:,} ({100*hvi_params/total_params:.1f}%)")
print(f"  DICAM_Feat: {dicam_feat_params:,} ({100*dicam_feat_params/total_params:.1f}%)")
print(f"  DICAM_Proj: {dicam_proj_params:,} ({100*dicam_proj_params/total_params:.1f}%)")

# 3. 前向测试 256x256
print("\n[3/4] 前向测试 256x256:")
model.eval()
torch.cuda.reset_peak_memory_stats()

with torch.no_grad():
    x_256 = torch.rand(1, 3, 256, 256).cuda()
    y_256 = model(x_256)

mem_256 = torch.cuda.max_memory_allocated() / 1024**3
has_nan_256 = torch.isnan(y_256).any().item()
has_inf_256 = torch.isinf(y_256).any().item()

print(f"  输出 shape: {y_256.shape}")
print(f"  输出范围: [{y_256.min().item():.4f}, {y_256.max().item():.4f}]")
print(f"  显存峰值: {mem_256:.3f} GB")
print(f"  NaN: {has_nan_256} | Inf: {has_inf_256}")

if has_nan_256 or has_inf_256:
    print("  ⚠️ 输出包含 NaN 或 Inf!")
    exit(1)

# 4. 前向测试 1024x1024
print("\n[4/4] 前向测试 1024x1024:")
torch.cuda.reset_peak_memory_stats()

with torch.no_grad():
    x_1024 = torch.rand(1, 3, 1024, 1024).cuda()
    y_1024 = model(x_1024)

mem_1024 = torch.cuda.max_memory_allocated() / 1024**3
has_nan_1024 = torch.isnan(y_1024).any().item()
has_inf_1024 = torch.isinf(y_1024).any().item()

print(f"  输出 shape: {y_1024.shape}")
print(f"  输出范围: [{y_1024.min().item():.4f}, {y_1024.max().item():.4f}]")
print(f"  显存峰值: {mem_1024:.3f} GB")
print(f"  NaN: {has_nan_1024} | Inf: {has_inf_1024}")

if has_nan_1024 or has_inf_1024:
    print("  ⚠️ 输出包含 NaN 或 Inf!")
    exit(1)

if mem_1024 >= 5.0:
    print(f"  ⚠️ 显存超过 5 GB ({mem_1024:.3f} GB)")

# 5. 总结
print("\n" + "=" * 70)
print("✅ 验证通过!")
print(f"  256x256 显存: {mem_256:.3f} GB")
print(f"  1024x1024 显存: {mem_1024:.3f} GB")
print(f"  参数量: {total_params:,} (HVI: {hvi_params:,}, DICAM: {dicam_feat_params+dicam_proj_params:,})")
print("=" * 70)
