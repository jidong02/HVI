#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
error_lumachroma.py  —  把误差能量拆成 消色(achromatic) vs 彩色(chromatic)

承接 error_psd.py 的结论(90% 误差在 LF)。LF 里混着两种东西:
  - 消色 LF: 照度/对比残差, 在 I 分支, ΔE 中性 -> 可能是免费 dB;
  - 彩色 LF: 色偏, 在 HV 分支, 与 ΔE 耦合 -> 你测过是 trade 区。
本脚本算: (1) 总误差能量里消色 vs 彩色各占多少;
          (2) 消色误差自身的 LF/MF/HF 频段分布。
据此判断: 那 90% 的 LF 误差到底是"I 分支可修的免费照度残差", 还是"颜色陷阱"。

分解依据: 误差向量 e=(eR,eG,eB) 沿灰轴 (1,1,1) 的投影 = 消色分量, 正交补 = 彩色分量。
          消色能量 = 3*mean(e)^2 求和; 彩色能量 = ||e||^2 - 消色能量 (Parseval, 正交可加)。

用法:
  python error_lumachroma.py --gt /path/UIEB/gt --pred /path/vanilla_out
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

EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]


def find_pairs(gt, pred):
    idx = lambda d: {os.path.splitext(f)[0]: os.path.join(d, f)
                     for f in os.listdir(d) if os.path.splitext(f)[1].lower() in EXTS}
    g, p = idx(gt), idx(pred)
    ks = sorted(set(g) & set(p))
    if not ks:
        raise RuntimeError(f"无匹配文件: {gt} vs {pred}")
    return [(g[k], p[k]) for k in ks]


def band_frac(x, lf=0.10, mf=0.25):
    """单通道误差的 LF/MF/HF 能量占比 (Hann 加窗, 归一化频率, Nyquist=0.5)。"""
    h, w = x.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    P = np.abs(np.fft.fftshift(np.fft.fft2(x * win))) ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(h))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(w))[None, :]
    r = np.sqrt(fy ** 2 + fx ** 2)
    t = P.sum() + 1e-12
    return (P[r < lf].sum() / t,
            P[(r >= lf) & (r < mf)].sum() / t,
            P[r >= mf].sum() / t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--lf", type=float, default=0.10)
    ap.add_argument("--mf", type=float, default=0.25)
    a = ap.parse_args()

    ach_share, ach_fr = [], []
    for gp, pp in find_pairs(a.gt, a.pred):
        gt, pr = _imread(gp), _imread(pp)
        if gt is None or pr is None:
            continue
        if gt.shape != pr.shape:
            pr = _resize(pr, gt.shape[0], gt.shape[1])
        e = (pr.astype(np.float64) - gt.astype(np.float64)) / 255.0  # [H,W,3]
        egray = e.mean(axis=2)                # 灰轴(消色)误差标量场
        tot = (e ** 2).sum() + 1e-12          # 总误差能量
        ach = (egray ** 2 * 3.0).sum()        # 沿灰轴投影能量
        ach_share.append(ach / tot)
        ach_fr.append(band_frac(egray, a.lf, a.mf))

    ach_share = np.array(ach_share)
    ach_fr = np.array(ach_fr)
    print(f"匹配图像数: {len(ach_share)}")
    print(f"消色 (achromatic / 照度) 误差能量占比: "
          f"{ach_share.mean()*100:6.2f}% +- {ach_share.std()*100:.2f}%")
    print(f"彩色 (chromatic / 色偏)   误差能量占比: "
          f"{(1-ach_share).mean()*100:6.2f}% +- {ach_share.std()*100:.2f}%")
    print(f"消色误差的频段分布 (mean):  "
          f"LF {ach_fr[:,0].mean()*100:5.2f}%   "
          f"MF {ach_fr[:,1].mean()*100:5.2f}%   "
          f"HF {ach_fr[:,2].mean()*100:5.2f}%")
    print("\n判读:")
    print("  消色占比高 + 消色误差也集中在 LF  => 90% 的 LF 误差主要是照度/对比残差,")
    print("     I 分支低频照度校正是对的方向, 且 ΔE 中性 (免费 dB)。")
    print("  彩色占比高                        => LF 误差以色偏为主, 落在你测过的颜色 trade 区。")


if __name__ == "__main__":
    main()