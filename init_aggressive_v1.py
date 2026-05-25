"""
合并 LPIPS baseline + DICAM 预训练 (aggressive v1 架构).

Aggressive v1 设计:
  - DICAM 作为特征提取器 (输出 256 通道)
  - 投影到 ch2 (36) 后作为残差注入到 HVE_block1 输出
  - HVI 主干输入保持原始 sRGB (无预处理)
"""
import torch
from pathlib import Path
from net.CIDNet import CIDNet

LPIPS_BASELINE = 'weights/baseline_lpips/epoch_35.pth'
DICAM_PRETRAINED = '/root/autodl-tmp/DICAM/ckpts/UIEB/DICAM_60.pt'
OUTPUT_DIR = 'weights/aggressive_v1'
OUTPUT_PATH = f'{OUTPUT_DIR}/init.pth'

print("=" * 70)
print("合并 LPIPS baseline + DICAM 预训练 (aggressive v1)")
print("=" * 70)

# 1. 创建新版 CIDNet (含 dicam_feat + dicam_proj)
model = CIDNet()
total_params = sum(p.numel() for p in model.parameters())
print(f"\n模型实例化完成, 总参数量: {total_params:,}")

# 参数量分解
hvi_params = sum(p.numel() for n, p in model.named_parameters() if 'dicam' not in n)
dicam_feat_params = sum(p.numel() for n, p in model.named_parameters() if 'dicam_feat' in n)
dicam_proj_params = sum(p.numel() for n, p in model.named_parameters() if 'dicam_proj' in n)
print(f"  HVI 主干: {hvi_params:,}")
print(f"  DICAM_Feat: {dicam_feat_params:,}")
print(f"  DICAM_Proj: {dicam_proj_params:,}")

# 2. 加载 LPIPS baseline 到 HVI 主干
print(f"\n[1/2] 加载 LPIPS baseline: {LPIPS_BASELINE}")
lpips_state = torch.load(LPIPS_BASELINE, map_location='cpu')

# 过滤掉旧的 dicam/fusion/awb keys
lpips_state_filtered = {k: v for k, v in lpips_state.items()
                        if not any(x in k for x in ['dicam', 'fusion', 'awb'])}
print(f"  过滤后保留 {len(lpips_state_filtered)} 个 keys (HVI 主干)")

missing, unexpected = model.load_state_dict(lpips_state_filtered, strict=False)
print(f"  Missing (应该是 dicam_feat.* + dicam_proj.*): {len(missing)}")
print(f"  Unexpected (应该为 0): {len(unexpected)}")

if unexpected:
    print(f"  ⚠️ Unexpected keys: {unexpected[:5]}")

# 3. 加载 DICAM 预训练到 dicam_feat (前 8 层)
print(f"\n[2/2] 加载 DICAM 预训练: {DICAM_PRETRAINED}")
dicam_raw = torch.load(DICAM_PRETRAINED, map_location='cpu')

# 处理可能的嵌套字典
if isinstance(dicam_raw, dict) and 'model_state_dict' in dicam_raw:
    dicam_state = dicam_raw['model_state_dict']
elif isinstance(dicam_raw, dict) and 'state_dict' in dicam_raw:
    dicam_state = dicam_raw['state_dict']
elif isinstance(dicam_raw, dict) and 'model' in dicam_raw:
    dicam_state = dicam_raw['model']
else:
    dicam_state = dicam_raw

# 去掉可能的 'module.' 前缀
if any(k.startswith('module.') for k in dicam_state.keys()):
    dicam_state = {k.replace('module.', ''): v for k, v in dicam_state.items()}

# 过滤掉 layer_tail (只要前 8 层)
dicam_state_filtered = {k: v for k, v in dicam_state.items() if 'layer_tail' not in k}
print(f"  过滤后保留 {len(dicam_state_filtered)} 个 keys (前 8 层)")

# 映射 key: layer_1_r.xxx -> dicam_feat.layer_1_r.xxx
dicam_state_mapped = {f'dicam_feat.{k}': v for k, v in dicam_state_filtered.items()}

missing_d, unexpected_d = model.load_state_dict(dicam_state_mapped, strict=False)
print(f"  Missing (应该只有 dicam_proj.*): {len(missing_d)}")
print(f"  Unexpected (应该为 0): {len(unexpected_d)}")

if missing_d:
    # 检查是否只剩 dicam_proj
    non_proj = [k for k in missing_d if 'dicam_proj' not in k]
    if non_proj:
        print(f"  ⚠️ 非 dicam_proj 的 missing keys: {non_proj[:5]}")

if unexpected_d:
    print(f"  ⚠️ Unexpected keys: {unexpected_d[:5]}")

# 4. 前向测试 + 数值检查
print("\n--- 前向测试 ---")
model.eval().cuda()
with torch.no_grad():
    x = torch.rand(2, 3, 256, 256).cuda()
    y = model(x)

print(f"Input:  {x.shape}, range=[{x.min().item():.3f}, {x.max().item():.3f}]")
print(f"Output: {y.shape}, range=[{y.min().item():.3f}, {y.max().item():.3f}]")
print(f"NaN: {torch.isnan(y).any().item()} | Inf: {torch.isinf(y).any().item()}")

# 5. 保存合并后的 checkpoint
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), OUTPUT_PATH)
print(f"\n✅ 合并后的 checkpoint 已保存: {OUTPUT_PATH}")
print(f"   大小: {Path(OUTPUT_PATH).stat().st_size / 1024 / 1024:.2f} MB")

# 6. 验证保存的 checkpoint 能被正确加载
print("\n--- 验证保存的 checkpoint ---")
model2 = CIDNet()
state2 = torch.load(OUTPUT_PATH, map_location='cpu')
missing2, unexpected2 = model2.load_state_dict(state2, strict=True)
print(f"Missing (应该为 0): {len(missing2)}")
print(f"Unexpected (应该为 0): {len(unexpected2)}")

print("\n" + "=" * 70)
print("✅ 完成! 现在可以用这个 checkpoint 启动训练:")
print(f"   --pretrain {OUTPUT_PATH}")
print("=" * 70)
