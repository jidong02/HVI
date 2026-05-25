"""
合并 LPIPS baseline + DICAM 官方预训练,生成 CIDNet+DICAM 的初始 checkpoint.
跑完后用生成的 checkpoint 作为训练的 --pretrain 起点.
"""
import torch
from pathlib import Path
from net.CIDNet import CIDNet

LPIPS_BASELINE = 'weights/baseline_lpips/epoch_35.pth'
DICAM_PRETRAINED = '/root/autodl-tmp/DICAM/ckpts/UIEB/DICAM_60.pt'
OUTPUT_DIR = 'weights/baseline_lpips_dicam'
OUTPUT_PATH = f'{OUTPUT_DIR}/init.pth'

print("=" * 60)
print("合并 LPIPS baseline + DICAM 预训练")
print("=" * 60)

# 1. 创建 CIDNet 模型(含未初始化的 dicam 和 fusion)
model = CIDNet()
print(f"\n模型实例化完成,参数量: {sum(p.numel() for p in model.parameters()):,}")

# 2. 加载 LPIPS baseline 到 HVI 主干
print(f"\n[1/2] 加载 LPIPS baseline: {LPIPS_BASELINE}")
lpips_state = torch.load(LPIPS_BASELINE, map_location='cpu')
missing, unexpected = model.load_state_dict(lpips_state, strict=False)
print(f"  Missing (应该全是 dicam.* 和 fusion.*): {len(missing)}")
print(f"  Unexpected (应该是 0): {len(unexpected)}")
if unexpected:
    print(f"  ⚠️ 意外的 keys: {unexpected[:5]}")

# 3. 加载 DICAM 预训练到 self.dicam
print(f"\n[2/2] 加载 DICAM 预训练: {DICAM_PRETRAINED}")
dicam_raw = torch.load(DICAM_PRETRAINED, map_location='cpu')

# 处理可能的嵌套字典
if isinstance(dicam_raw, dict) and 'state_dict' in dicam_raw:
    dicam_state = dicam_raw['state_dict']
elif isinstance(dicam_raw, dict) and 'model' in dicam_raw:
    dicam_state = dicam_raw['model']
else:
    dicam_state = dicam_raw

# 去掉可能的 'module.' 前缀(多卡训练保留下来的)
if any(k.startswith('module.') for k in dicam_state.keys()):
    dicam_state = {k.replace('module.', ''): v for k, v in dicam_state.items()}

# 加载到 model.dicam (注意 model.dicam.load_state_dict 不是 model.load_state_dict)
missing_d, unexpected_d = model.dicam.load_state_dict(dicam_state, strict=False)
print(f"  DICAM Missing: {len(missing_d)}")
print(f"  DICAM Unexpected: {len(unexpected_d)}")
if missing_d or unexpected_d:
    print(f"  ⚠️ DICAM 加载有问题:")
    for k in missing_d[:3]: print(f"    Missing: {k}")
    for k in unexpected_d[:3]: print(f"    Unexpected: {k}")
    print("  → 可能 DICAM 官方权重的 key 命名不匹配,需要调整")

# 4. 前向测试 + 数值检查
print("\n--- 前向测试 ---")
model.eval().cuda()
with torch.no_grad():
    x = torch.rand(2, 3, 256, 256).cuda()
    y = model(x)
print(f"Input:  {x.shape}, range=[{x.min().item():.3f}, {x.max().item():.3f}]")
print(f"Output: {y.shape}, range=[{y.min().item():.3f}, {y.max().item():.3f}]")
print(f"NaN: {torch.isnan(y).any().item()} | Inf: {torch.isinf(y).any().item()}")

# 5. 保存合并后的初始 checkpoint
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), OUTPUT_PATH)
print(f"\n✅ 合并后的初始 checkpoint 已保存: {OUTPUT_PATH}")
print(f"   大小: {Path(OUTPUT_PATH).stat().st_size / 1024 / 1024:.2f} MB")

# 6. 验证保存的 checkpoint 能被正确加载
print("\n--- 验证保存的 checkpoint ---")
model2 = CIDNet()
state2 = torch.load(OUTPUT_PATH, map_location='cpu')
missing2, unexpected2 = model2.load_state_dict(state2, strict=False)
print(f"Missing (应该为 0): {len(missing2)}")
print(f"Unexpected (应该为 0): {len(unexpected2)}")

print("\n" + "=" * 60)
print("✅ 完成!现在可以用这个 checkpoint 启动训练:")
print(f"   --pretrain {OUTPUT_PATH}")
print("=" * 60)