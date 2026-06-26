#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
error_psd.py  —  误差功率谱诊断 (Where does the recoverable error live?)

理论依据 (Parseval): PSNR = 10*log10(MAX^2 / MSE), 而 MSE 即 L2 误差能量;
由 Parseval, 空域误差能量 == 频域误差能量在所有频段上的积分。
=> 残差功率谱中能量最大的频段, 就是"可恢复 dB"所在地。

本脚本计算 (pred - gt) 的**径向平均功率谱 (radially-averaged PSD)**, 用于:
  (1) 动机图: 可恢复误差落在哪个频段 -> 为什么走/不走某类频域模块;
  (2) 机制图: vanilla vs 你的模块, 看模块是否削掉了它该打的那段能量。

严谨性内置:
  - PSD 默认算在 RGB(与 PSNR 同域, Parseval->PSNR 精确), 同时给 luma(Rec.601)视角;
  - FFT 前加 2D Hann 窗, 抑制图像边界泄漏造成的假高频;
  - 频率轴用归一化 cycles/pixel (Nyquist=0.5), 不同尺寸图像可直接聚合;
  - 全测试集聚合, 报 LF/MF/HF 三段能量占比 (mean +- std);
  - 打印各方法平均 PSNR 作为 sanity check (总能量应与 1/10^(PSNR/10) 同向)。

用法:
  python error_psd.py --gt /path/to/gt --pred /path/to/vanilla_out \
      [--pred2 /path/to/module_out] [--label vanilla] [--label2 ours] \
      [--out ./error_psd.png] [--bins 128] [--domain rgb|luma|both]

输入: 三个文件夹里文件名 stem 相同 (扩展名可不同)。pred/pred2 为某方法的增强输出。
"""
import os
import argparse
import numpy as np

try:
    import cv2
    def _imread(p):
        img = cv2.imread(p, cv2.IMREAD_COLOR)        # BGR, uint8
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # -> RGB
except Exception:
    from PIL import Image
    def _imread(p):
        try:
            return np.array(Image.open(p).convert("RGB"))
        except Exception:
            return None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


IMG_EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]


def find_pairs(gt_dir, pred_dir):
    """按文件名 stem 匹配 gt 与 pred。"""
    def index(d):
        m = {}
        for f in os.listdir(d):
            stem, ext = os.path.splitext(f)
            if ext.lower() in IMG_EXTS:
                m[stem] = os.path.join(d, f)
        return m
    g, p = index(gt_dir), index(pred_dir)
    keys = sorted(set(g) & set(p))
    if not keys:
        raise RuntimeError(f"无匹配文件: {gt_dir} vs {pred_dir} (检查文件名 stem 是否一致)")
    return [(k, g[k], p[k]) for k in keys]


def to_luma(rgb):
    # Rec.601 luma, 与多数 UIE 论文报 Y-PSNR 时一致
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def hann2d(h, w):
    return np.outer(np.hanning(h), np.hanning(w))


def radial_psd(err2d, bins, fmax=0.5):
    """单通道 2D 误差 -> 归一化频率上的径向平均功率谱。
    err2d: float, 值域 [0,1] (尺度只是单调缩放, 不影响频段排序/占比)。
    返回: centers(归一化频率, cycles/pixel), psd(每个频率环的平均功率)。
    """
    h, w = err2d.shape
    win = hann2d(h, w)
    F = np.fft.fftshift(np.fft.fft2(err2d * win))
    P = (np.abs(F) ** 2) / (h * w)                      # 功率, 归一化
    fy = np.fft.fftshift(np.fft.fftfreq(h))[:, None]    # [-0.5, 0.5)
    fx = np.fft.fftshift(np.fft.fftfreq(w))[None, :]
    r = np.sqrt(fy ** 2 + fx ** 2)                      # 归一化径向频率
    edges = np.linspace(0.0, fmax, bins + 1)
    idx = np.digitize(r.ravel(), edges) - 1
    valid = (idx >= 0) & (idx < bins)
    psd = np.bincount(idx[valid], weights=P.ravel()[valid], minlength=bins)
    cnt = np.bincount(idx[valid], minlength=bins).astype(np.float64)
    psd = psd / np.maximum(cnt, 1.0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, psd


def band_fractions(err2d, lf=0.1, mf=0.25):
    """直接在 2D 功率图上按掩膜求三段能量占比 (精确的 Parseval 分解)。"""
    h, w = err2d.shape
    win = hann2d(h, w)
    F = np.fft.fftshift(np.fft.fft2(err2d * win))
    P = (np.abs(F) ** 2)
    fy = np.fft.fftshift(np.fft.fftfreq(h))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(w))[None, :]
    r = np.sqrt(fy ** 2 + fx ** 2)
    tot = P.sum() + 1e-12
    e_lf = P[r < lf].sum() / tot
    e_mf = P[(r >= lf) & (r < mf)].sum() / tot
    e_hf = P[r >= mf].sum() / tot
    return e_lf, e_mf, e_hf


def psnr_rgb(pred, gt):
    mse = np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2)
    if mse <= 1e-12:
        return 99.0
    return 10.0 * np.log10((255.0 ** 2) / mse)


def analyze(gt_dir, pred_dir, bins, domain, lf, mf):
    pairs = find_pairs(gt_dir, pred_dir)
    psd_rgb_acc, psd_lum_acc = [], []
    frac_acc = []
    psnrs = []
    centers = None
    n = 0
    for k, gpath, ppath in pairs:
        gt = _imread(gpath)
        pr = _imread(ppath)
        if gt is None or pr is None:
            print(f"  [skip] 读取失败: {k}")
            continue
        if gt.shape != pr.shape:
            # 尺寸不一致则把 pred resize 到 gt (评测时本应一致, 仅兜底)
            pr = cv2.resize(pr, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_CUBIC)
        psnrs.append(psnr_rgb(pr, gt))
        prf = pr.astype(np.float64) / 255.0
        gtf = gt.astype(np.float64) / 255.0
        err = prf - gtf                                  # [H,W,3]

        if domain in ("rgb", "both"):
            ps = []
            for c in range(3):
                cen, p = radial_psd(err[..., c], bins)
                ps.append(p)
            centers = cen
            psd_rgb_acc.append(np.mean(ps, axis=0))
            # 三段占比也在 RGB 上 (逐通道求后平均)
            fr = np.mean([band_fractions(err[..., c], lf, mf) for c in range(3)], axis=0)
            frac_acc.append(fr)
        if domain in ("luma", "both"):
            lerr = to_luma(prf) - to_luma(gtf)
            cen, p = radial_psd(lerr, bins)
            centers = cen
            psd_lum_acc.append(p)
            if domain == "luma":
                frac_acc.append(band_fractions(lerr, lf, mf))
        n += 1

    out = {
        "n": n,
        "centers": centers,
        "psnr_mean": float(np.mean(psnrs)) if psnrs else float("nan"),
        "psd_rgb": np.mean(psd_rgb_acc, axis=0) if psd_rgb_acc else None,
        "psd_lum": np.mean(psd_lum_acc, axis=0) if psd_lum_acc else None,
        "frac_mean": np.mean(frac_acc, axis=0) if frac_acc else None,
        "frac_std": np.std(frac_acc, axis=0) if frac_acc else None,
    }
    return out


def report(label, res, lf, mf):
    print(f"\n=== {label} ===")
    print(f"  匹配图像数: {res['n']}   平均 PSNR(RGB): {res['psnr_mean']:.3f} dB")
    if res["frac_mean"] is not None:
        fm, fs = res["frac_mean"], res["frac_std"]
        print(f"  误差能量占比 (mean +- std):")
        print(f"    LF  (f<{lf}):         {fm[0]*100:6.2f}% +- {fs[0]*100:.2f}%")
        print(f"    MF  ({lf}<=f<{mf}):   {fm[1]*100:6.2f}% +- {fs[1]*100:.2f}%")
        print(f"    HF  (f>={mf}):        {fm[2]*100:6.2f}% +- {fs[2]*100:.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="GT 文件夹")
    ap.add_argument("--pred", required=True, help="方法1输出文件夹 (如 vanilla)")
    ap.add_argument("--pred2", default=None, help="方法2输出文件夹 (如 ours), 可选")
    ap.add_argument("--label", default="method1")
    ap.add_argument("--label2", default="method2")
    ap.add_argument("--out", default="./error_psd.png")
    ap.add_argument("--bins", type=int, default=128)
    ap.add_argument("--domain", choices=["rgb", "luma", "both"], default="both")
    ap.add_argument("--lf", type=float, default=0.10, help="LF/MF 分界 (cycles/pixel)")
    ap.add_argument("--mf", type=float, default=0.25, help="MF/HF 分界 (cycles/pixel)")
    args = ap.parse_args()

    print("分析方法1 ...")
    r1 = analyze(args.gt, args.pred, args.bins, args.domain, args.lf, args.mf)
    report(args.label, r1, args.lf, args.mf)

    r2 = None
    if args.pred2:
        print("\n分析方法2 ...")
        r2 = analyze(args.gt, args.pred2, args.bins, args.domain, args.lf, args.mf)
        report(args.label2, r2, args.lf, args.mf)

    # ---- 画图 ----
    use_rgb = args.domain in ("rgb", "both")
    cols = 1 + (1 if r2 is not None else 0)
    fig, axes = plt.subplots(1, cols, figsize=(6.2 * cols, 4.6))
    if cols == 1:
        axes = [axes]

    key = "psd_rgb" if use_rgb else "psd_lum"
    dom_name = "RGB" if use_rgb else "luma"

    ax = axes[0]
    ax.semilogy(r1["centers"], r1[key], lw=2, label=args.label)
    if r2 is not None:
        ax.semilogy(r2["centers"], r2[key], lw=2, label=args.label2)
    ax.axvline(args.lf, color="gray", ls="--", lw=0.8)
    ax.axvline(args.mf, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("normalized spatial frequency (cycles/pixel)")
    ax.set_ylabel(f"error power ({dom_name}, log)")
    ax.set_title("Radially-averaged error PSD")
    ax.legend()
    ax.grid(alpha=0.3)

    if r2 is not None:
        ax2 = axes[1]
        diff = r1[key] - r2[key]   # >0 表示方法2在该频段削掉了误差能量
        ax2.plot(r1["centers"], diff, lw=2, color="crimson")
        ax2.axhline(0, color="k", lw=0.8)
        ax2.axvline(args.lf, color="gray", ls="--", lw=0.8)
        ax2.axvline(args.mf, color="gray", ls="--", lw=0.8)
        ax2.set_xlabel("normalized spatial frequency (cycles/pixel)")
        ax2.set_ylabel(f"PSD({args.label}) - PSD({args.label2})")
        ax2.set_title("Where method2 reduces error (>0 = reduced)")
        ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"\n图已保存: {args.out}")
    if args.domain == "both":
        print("提示: 已用 RGB 视角画图(与 PSNR 同域);luma 视角的曲线数据也已计算,"
              "需要的话可改 key='psd_lum' 再出一张。")


if __name__ == "__main__":
    main()