#!/usr/bin/env python3
"""
HVI-CIDNet 训练可视化工具 (W&B 简易替代品)

放置位置: /root/autodl-tmp/HVI-CIDNet/plot_run.py

功能:
  ✅ 自动找最新的 metrics.md
  ✅ 绘制 PSNR / SSIM / LPIPS 三联折线图
  ✅ baseline 参考线 + 最佳 epoch 高亮
  ✅ 多 run 对比（叠加在同一张图）
  ✅ 导出 PNG + CSV + 文本摘要
  ✅ 可被 train.py 直接 import 调用（训练完自动出图）

用法 1: 自动模式（画最新一次训练）
    python plot_run.py

用法 2: 指定文件
    python plot_run.py --runs results/training/metrics2026-05-21-181534.md

用法 3: 多 run 对比
    python plot_run.py \\
        --runs results/training/metrics_baseline.md \\
               results/training/metrics_lpips.md \\
               results/training/metrics_wacc.md \\
        --labels "Baseline" "+ LPIPS" "+ LPIPS + WACC"

用法 4: 改 baseline 数值
    python plot_run.py --baseline-psnr 21.6797 --baseline-ssim 0.8917 --baseline-lpips 0.167

用法 5: 在 train.py 里 import 调用（见文末 INTEGRATE 部分）
"""

import argparse
import glob
import re
import sys
from datetime import datetime
from pathlib import Path


# ============== Baseline 默认值（你的 HVI-CIDNet epoch_40 finetune）==============
DEFAULT_BASELINE = {
    'PSNR':  21.6797,
    'SSIM':  0.8917,
    'LPIPS': 0.167,
}


# ============== Run 配色（按顺序循环使用）==============
RUN_COLORS = [
    '#185FA5',   # 蓝
    '#0F6E56',   # 青
    '#993C1D',   # 红棕
    '#534AB7',   # 紫
    '#854F0B',   # 棕
]


def parse_metrics_md(path):
    """
    解析 train.py 生成的 metrics.md 文件.

    返回:
      {
        'header': {'dataset': 'uieb', 'lr': '0.0001', ...},
        'epochs': [5, 10, ...],
        'PSNR':   [21.57, 21.58, ...],
        'SSIM':   [0.893, 0.892, ...],
        'LPIPS':  [0.144, 0.142, ...],
        'path': '...',
      }
    """
    path = Path(path)
    text = path.read_text()
    lines = text.splitlines()

    # 解析 header (key: value 行)
    header = {}
    in_table = False
    for line in lines:
        if line.startswith('|'):
            in_table = True
            break
        if ':' in line:
            k, v = line.split(':', 1)
            header[k.strip()] = v.strip()

    # 解析表格 (| epoch | PSNR | SSIM | LPIPS |)
    epochs, psnr, ssim, lpips = [], [], [], []
    for line in lines:
        if not line.startswith('|'):
            continue
        cells = [c.strip() for c in line.strip('|').split('|')]
        if len(cells) < 4:
            continue
        # 跳过表头和分隔行
        if any(s in cells[0].lower() for s in ['epoch', '---']):
            continue
        try:
            epochs.append(int(cells[0]))
            psnr.append(float(cells[1]))
            ssim.append(float(cells[2]))
            lpips.append(float(cells[3]))
        except (ValueError, IndexError):
            continue

    return {
        'header': header,
        'epochs': epochs,
        'PSNR': psnr,
        'SSIM': ssim,
        'LPIPS': lpips,
        'path': str(path),
    }


def find_latest_metrics(metrics_dir='results/training'):
    """找最新的 metrics*.md 文件"""
    candidates = glob.glob(str(Path(metrics_dir) / 'metrics*.md'))
    if not candidates:
        return None
    return max(candidates, key=lambda p: Path(p).stat().st_mtime)


def auto_label(run, fallback='run'):
    """从 header 自动生成 run 标签"""
    h = run['header']
    parts = []
    if 'dataset' in h:
        parts.append(h['dataset'])
    if 'LPIPS_weight' in h and float(h.get('LPIPS_weight', 0)) > 0:
        parts.append(f"LPIPS={h['LPIPS_weight']}")
    if 'LAB_weight' in h and float(h.get('LAB_weight', 0)) > 0:
        parts.append(f"LAB={h['LAB_weight']}")
    if not parts:
        return fallback
    return ' | '.join(parts)


def plot_runs(runs, labels, baseline, out_path, title=None):
    """
    画三联图: PSNR / SSIM / LPIPS.

    runs: list of parsed run dicts
    labels: list of strings, same length as runs
    baseline: dict {'PSNR': ..., 'SSIM': ..., 'LPIPS': ...} 或 None
    """
    # 延迟导入 matplotlib (避免脚本启动时就 import)
    import matplotlib
    matplotlib.use('Agg')  # 无 GUI 环境
    import matplotlib.pyplot as plt
    import numpy as np

    metrics_info = [
        ('PSNR',  'PSNR ↑ (dB)', 'higher is better', np.argmax),
        ('SSIM',  'SSIM ↑',      'higher is better', np.argmax),
        ('LPIPS', 'LPIPS ↓',     'lower is better',  np.argmin),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=False)
    fig.subplots_adjust(hspace=0.32)

    GRAY = '#5F5E5A'
    CORAL = '#D85A30'

    for ax, (metric, ylabel, hint, argbest_fn) in zip(axes, metrics_info):
        # 画 baseline 横线
        if baseline and metric in baseline:
            bval = baseline[metric]
            ax.axhline(bval, color=GRAY, linestyle='--', linewidth=1.5,
                       label=f'Baseline = {bval:.4f}', zorder=2, alpha=0.8)

        # 画各个 run
        for i, (run, label) in enumerate(zip(runs, labels)):
            color = RUN_COLORS[i % len(RUN_COLORS)]
            epochs = run['epochs']
            vals = run[metric]
            if not vals:
                continue

            ax.plot(epochs, vals, 'o-', color=color, linewidth=2,
                    markersize=6, markerfacecolor='white',
                    markeredgecolor=color, markeredgewidth=1.5,
                    label=label, zorder=3)

            # 最佳 epoch 星标 (每个 run 一个)
            best_idx = argbest_fn(vals)
            best_ep = epochs[best_idx]
            best_val = vals[best_idx]
            ax.plot(best_ep, best_val, '*', color=color, markersize=18,
                    markeredgecolor='white', markeredgewidth=1.0, zorder=5)
            # 标注最佳值
            offset = (max(vals) - min(vals)) * 0.08
            if metric == 'LPIPS':
                ytxt = best_val - offset
                va = 'top'
            else:
                ytxt = best_val + offset
                va = 'bottom'
            ax.annotate(f'  ep{best_ep}: {best_val:.4f}',
                        xy=(best_ep, best_val), xytext=(best_ep, ytxt),
                        ha='center', va=va, fontsize=8,
                        color=color, fontweight='bold')

        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xlabel('Epoch', fontsize=10)
        ax.text(0.99, 0.02, hint, transform=ax.transAxes,
                ha='right', va='bottom', fontsize=8, color='gray', style='italic')
        ax.grid(True, alpha=0.25, linestyle=':')
        ax.legend(loc='best', fontsize=9, framealpha=0.95, edgecolor='lightgray')
        ax.tick_params(labelsize=9)

    if not title:
        title = 'HVI-CIDNet Training Metrics'
    fig.suptitle(title, fontsize=13, fontweight='bold', y=0.995)

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


def export_csv(runs, labels, out_path):
    """导出 CSV，方便论文里直接用"""
    import csv
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['run', 'epoch', 'PSNR', 'SSIM', 'LPIPS'])
        for run, label in zip(runs, labels):
            for i, ep in enumerate(run['epochs']):
                w.writerow([label, ep, run['PSNR'][i], run['SSIM'][i], run['LPIPS'][i]])


def print_summary(runs, labels, baseline=None):
    """打印训练摘要表"""
    import numpy as np

    print('\n' + '=' * 72)
    print(' Training Summary')
    print('=' * 72)

    if baseline:
        print(f' Baseline:  PSNR={baseline["PSNR"]:.4f}  '
              f'SSIM={baseline["SSIM"]:.4f}  LPIPS={baseline["LPIPS"]:.4f}')
        print('-' * 72)

    header = f' {"Run":<28} {"Best PSNR":>12} {"Best SSIM":>12} {"Best LPIPS":>12}'
    print(header)
    print('-' * 72)

    for run, label in zip(runs, labels):
        if not run['epochs']:
            continue
        best_psnr_idx = int(np.argmax(run['PSNR']))
        best_ssim_idx = int(np.argmax(run['SSIM']))
        best_lpips_idx = int(np.argmin(run['LPIPS']))

        psnr_str  = f"{run['PSNR'][best_psnr_idx]:.4f} (ep{run['epochs'][best_psnr_idx]})"
        ssim_str  = f"{run['SSIM'][best_ssim_idx]:.4f} (ep{run['epochs'][best_ssim_idx]})"
        lpips_str = f"{run['LPIPS'][best_lpips_idx]:.4f} (ep{run['epochs'][best_lpips_idx]})"

        print(f' {label[:28]:<28} {psnr_str:>12} {ssim_str:>12} {lpips_str:>12}')

        if baseline:
            d_psnr  = run['PSNR'][best_psnr_idx]  - baseline['PSNR']
            d_ssim  = run['SSIM'][best_ssim_idx]  - baseline['SSIM']
            d_lpips = (run['LPIPS'][best_lpips_idx] - baseline['LPIPS']) / baseline['LPIPS'] * 100
            print(f' {"  Δ vs baseline":<28} {d_psnr:>+11.4f}  {d_ssim:>+11.4f}  {d_lpips:>+10.1f}%')
    print('=' * 72 + '\n')


def plot_from_metrics_file(metrics_path, baseline=None, out_dir='results/plots'):
    """
    便捷函数: 给 train.py 直接 import 用.
    传一个 metrics.md 文件路径,自动生成 PNG/CSV/Summary.
    """
    run = parse_metrics_md(metrics_path)
    label = auto_label(run)
    if baseline is None:
        baseline = DEFAULT_BASELINE
    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    out_png = Path(out_dir) / f'run_{ts}.png'
    out_csv = Path(out_dir) / f'run_{ts}.csv'
    plot_runs([run], [label], baseline, out_png,
              title=f'Run: {Path(metrics_path).name}')
    export_csv([run], [label], out_csv)
    print_summary([run], [label], baseline)
    print(f'📊 Chart saved to: {out_png}')
    print(f'📄 CSV saved to:   {out_csv}')
    return out_png, out_csv


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter,
                                description=__doc__)
    p.add_argument('--runs', nargs='+', default=None,
                   help='一个或多个 metrics.md 路径。不指定则用最新的。')
    p.add_argument('--labels', nargs='+', default=None,
                   help='对应每个 run 的标签。不指定则从 header 自动生成。')
    p.add_argument('--metrics-dir', default='results/training',
                   help='查找 metrics.md 的目录 (默认: results/training)')
    p.add_argument('--out-dir', default='results/plots',
                   help='输出目录 (默认: results/plots)')
    p.add_argument('--baseline-psnr', type=float, default=DEFAULT_BASELINE['PSNR'])
    p.add_argument('--baseline-ssim', type=float, default=DEFAULT_BASELINE['SSIM'])
    p.add_argument('--baseline-lpips', type=float, default=DEFAULT_BASELINE['LPIPS'])
    p.add_argument('--no-baseline', action='store_true',
                   help='不画 baseline 参考线')
    p.add_argument('--title', default=None)
    args = p.parse_args()

    # 找 runs
    if args.runs is None:
        latest = find_latest_metrics(args.metrics_dir)
        if latest is None:
            print(f'❌ 在 {args.metrics_dir} 没找到 metrics*.md 文件')
            sys.exit(1)
        print(f'📂 Auto-detected latest: {latest}')
        run_paths = [latest]
    else:
        run_paths = args.runs

    # 解析所有 runs
    runs = [parse_metrics_md(p) for p in run_paths]

    # 生成标签
    if args.labels:
        if len(args.labels) != len(runs):
            print(f'⚠️ labels 数量({len(args.labels)}) 与 runs 数量({len(runs)}) 不一致')
            sys.exit(1)
        labels = args.labels
    else:
        labels = [auto_label(r, fallback=Path(p).stem)
                  for r, p in zip(runs, run_paths)]

    # baseline
    baseline = None if args.no_baseline else {
        'PSNR':  args.baseline_psnr,
        'SSIM':  args.baseline_ssim,
        'LPIPS': args.baseline_lpips,
    }

    # 输出文件名
    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    out_png = Path(args.out_dir) / f'compare_{ts}.png'
    out_csv = Path(args.out_dir) / f'compare_{ts}.csv'

    # 画图 + 导出
    plot_runs(runs, labels, baseline, out_png, title=args.title)
    export_csv(runs, labels, out_csv)
    print_summary(runs, labels, baseline)
    print(f'📊 Chart saved to: {out_png}')
    print(f'📄 CSV saved to:   {out_csv}')


if __name__ == '__main__':
    main()


# ====================================================================
# INTEGRATE: 在 train.py 训练完后自动调用 (可选)
# --------------------------------------------------------------------
# 在 train.py 最末尾 (最外层 for epoch loop 结束后) 加这几行:
#
#     # 训练完自动生成可视化
#     try:
#         from plot_run import plot_from_metrics_file
#         plot_from_metrics_file(f"./results/training/metrics{now}.md")
#     except Exception as e:
#         print(f'⚠️ Auto-plot failed: {e}')
# ====================================================================