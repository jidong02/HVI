#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
output_psd.py  —  输出谱 vs GT 谱 (真恢复 还是 放噪?)

证据用途(机制桥之一, 不暴露模块结构):
  - 输出谱在 HF 段贴合 GT  => 恢复的是真实结构;
  - 输出谱在 HF 段超出 GT (过冲) => 引入了 GT 中不存在的能量 = 噪声放大 (naive wavelet 的签名)。
与 error_psd.py 同构, 区别: 这里算**图像本身**的径向平均功率谱, 而非误差谱。

输出:
  - 一张图: GT 谱 + 各方法谱 叠加 (log-y, 归一化频率, Hann 加窗, 全集聚合);
  - 打印各方法在 HF 段 (f>=mf) 相对 GT 的能量比: >1 过冲(放噪), ~1 贴合(真恢复), <1 欠恢复。

用法 (方法文件夹用逗号分隔, labels 一一对应):
  python output_psd.py --gt /path/UIEB/gt \
      --preds /path/vanilla_out,/path/ours_out --labels vanilla,ours \
      [--preds .../wavelet_out 也可三路对比] [--domain luma|rgb] [--out ./output_psd.png]
"""
import os
import argparse
import numpy as np

try:
    import cv2
    def _imread(p):
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        return None if im is None else cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    def _resize(im, h, w):
        return cv2.resize(im, (w, h), interpolation=cv2.INTER_CUBIC)
except Exception:
    from PIL import Image
    def _imread(p):
        try:
            return np.array(Image.open(p).convert("RGB"))
        except Exception:
            return None
    def _resize(im, h, w):
        return np.array(Image.fromarray(im).resize((w, h), Image.BICUBIC))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]


def index(d):
    return {os.path.splitext(f)[0]: os.path.join(d, f)
            for f in os.listdir(d) if os.path.splitext(f)[1].lower() in EXTS}


def to_luma(rgb):
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def radial_psd(x2d, bins, fmax=0.5):
    """单通道图像 -> 归一化频率上的径向平均功率谱 (Hann 加窗)。"""
    h, w = x2d.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    F = np.fft.fftshift(np.fft.fft2(x2d * win))
    P = (np.abs(F) ** 2) / (h * w)
    fy = np.fft.fftshift(np.fft.fftfreq(h))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(w))[None, :]
    r = np.sqrt(fy ** 2 + fx ** 2)
    edges = np.linspace(0.0, fmax, bins + 1)
    idx = np.digitize(r.ravel(), edges) - 1
    valid = (idx >= 0) & (idx < bins)
    psd = np.bincount(idx[valid], weights=P.ravel()[valid], minlength=bins)
    cnt = np.bincount(idx[valid], minlength=bins).astype(np.float64)
    psd = psd / np.maximum(cnt, 1.0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, psd


def mean_psd(folder, gt_idx, bins, domain):
    """对 folder 内与 gt 匹配的图, 算 (方法谱, 对应GT谱) 的均值。"""
    f_idx = index(folder)
    keys = sorted(set(f_idx) & set(gt_idx))
    if not keys:
        raise RuntimeError(f"无匹配: {folder}")
    sp, sg, centers = [], [], None
    for k in keys:
        pr, gt = _imread(f_idx[k]), _imread(gt_idx[k])
        if pr is None or gt is None:
            continue
        if pr.shape != gt.shape:
            pr = _resize(pr, gt.shape[0], gt.shape[1])
        prf, gtf = pr.astype(np.float64) / 255.0, gt.astype(np.float64) / 255.0
        if domain == "rgb":
            pp = np.mean([radial_psd(prf[..., c], bins)[1] for c in range(3)], axis=0)
            cen, _ = radial_psd(prf[..., 0], bins)
            gg = np.mean([radial_psd(gtf[..., c], bins)[1] for c in range(3)], axis=0)
        else:
            cen, pp = radial_psd(to_luma(prf), bins)
            _, gg = radial_psd(to_luma(gtf), bins)
        centers = cen
        sp.append(pp)
        sg.append(gg)
    return centers, np.mean(sp, axis=0), np.mean(sg, axis=0), len(sp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--preds", required=True, help="逗号分隔的多个方法输出目录")
    ap.add_argument("--labels", required=True, help="逗号分隔, 与 --preds 一一对应")
    ap.add_argument("--out", default="./output_psd.png")
    ap.add_argument("--bins", type=int, default=128)
    ap.add_argument("--domain", choices=["luma", "rgb"], default="luma")
    ap.add_argument("--mf", type=float, default=0.25, help="HF 段下界 (cycles/pixel)")
    a = ap.parse_args()

    preds = [s.strip() for s in a.preds.split(",") if s.strip()]
    labels = [s.strip() for s in a.labels.split(",") if s.strip()]
    assert len(preds) == len(labels), "--preds 与 --labels 数量必须一致"

    gt_idx = index(a.gt)
    gt_curve, centers = None, None
    curves = []
    for folder, lab in zip(preds, labels):
        cen, p_psd, g_psd, n = mean_psd(folder, gt_idx, a.bins, a.domain)
        centers = cen
        if gt_curve is None:
            gt_curve = g_psd            # GT 谱以第一个方法的匹配集为准
        curves.append((lab, p_psd, n))

    # HF 能量比 (相对 GT): >1 过冲(放噪), ~1 贴合, <1 欠恢复
    hf = centers >= a.mf
    gt_hf = gt_curve[hf].sum() + 1e-12
    print(f"域: {a.domain}    HF 段定义: f >= {a.mf}")
    print(f"{'方法':<16}{'匹配数':>8}{'HF能量/GT_HF':>16}   判读")
    for lab, p, n in curves:
        ratio = p[hf].sum() / gt_hf
        verdict = "过冲(放噪信号)" if ratio > 1.10 else ("欠恢复" if ratio < 0.90 else "贴合GT(真恢复)")
        print(f"{lab:<16}{n:>8}{ratio:>16.3f}   {verdict}")

    # 画图
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.semilogy(centers, gt_curve, lw=2.5, color="k", ls="--", label="GT")
    for lab, p, _ in curves:
        ax.semilogy(centers, p, lw=2, label=lab)
    ax.axvline(a.mf, color="gray", ls=":", lw=0.9)
    ax.set_xlabel("normalized spatial frequency (cycles/pixel)")
    ax.set_ylabel(f"power spectrum ({a.domain}, log)")
    ax.set_title("Output spectrum vs GT  (HF overshoot = noise amplification)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(a.out, dpi=200)
    print(f"\n图已保存: {a.out}")
    print("读图: 看 HF 段(竖线右侧)各方法谱相对黑色 GT 虚线 —— 贴合=真恢复, 高出=过冲放噪。")


if __name__ == "__main__":
    main()