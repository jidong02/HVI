#!/usr/bin/env python3
"""Evaluate one checkpoint: PSNR, dE2000, UIQM, UCIQE.
Usage: python eval_colors.py <weight.pth> <low_dir> <gt_dir> [--device cuda|cpu]"""
import os, sys, glob, argparse
import torch, torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from skimage.color import rgb2lab, deltaE_ciede2000
from metrics.no_reference import calc_uiqm, calc_uciqe


def calc_psnr(arr1, arr2):
    diff = arr1.astype(np.float32) - arr2.astype(np.float32)
    return 10.0 * np.log10(255.0 * 255.0 / (np.mean(diff ** 2) + 1e-8))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('weight', help='pth checkpoint')
    parser.add_argument('low_dir', help='test low input dir')
    parser.add_argument('gt_dir', help='test GT/high dir')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--known-psnr', type=float, default=None)
    args = parser.parse_args()

    device = args.device if args.device == 'cuda' and torch.cuda.is_available() else 'cpu'

    from net.CIDNet import CIDNet
    model = CIDNet().to(device)
    sd = torch.load(args.weight, map_location=device)
    model.load_state_dict(sd, strict=False)
    model.eval()
    print(f'Loaded: {args.weight}')

    files = sorted(glob.glob(os.path.join(args.low_dir, '*')))
    os.makedirs('/tmp/eval_out', exist_ok=True)
    for old in glob.glob('/tmp/eval_out/*'):
        os.remove(old)

    factor = 8
    for f in tqdm(files, desc='infer'):
        name = os.path.basename(f)
        img = Image.open(f).convert('RGB')
        x = transforms.ToTensor()(img).unsqueeze(0).to(device)
        _, _, h, w = x.shape
        H = ((h + factor) // factor) * factor
        W = ((w + factor) // factor) * factor
        padh, padw = H - h, W - w
        if padh or padw:
            x = F.pad(x, (0, padw, 0, padh), 'reflect')
        with torch.inference_mode():
            out = model(x)
        out = out[:, :, :h, :w].clamp(0, 1)
        out_img = transforms.ToPILImage()(out.squeeze(0).cpu())
        out_img.save(f'/tmp/eval_out/{name}')
    torch.cuda.empty_cache()

    psnr_sum, de_sum, uiqm_sum, uciqe_sum, n = 0.0, 0.0, 0.0, 0.0, 0
    for out_f in tqdm(sorted(glob.glob('/tmp/eval_out/*')), desc='metrics'):
        name = os.path.basename(out_f)
        gt_f = os.path.join(args.gt_dir, name)
        if not os.path.exists(gt_f):
            continue
        enh = np.array(Image.open(out_f).convert('RGB'))
        gt = np.array(Image.open(gt_f).convert('RGB'))
        psnr_sum += calc_psnr(enh, gt)
        lab_enh = rgb2lab(enh / 255.0)
        lab_gt = rgb2lab(gt / 255.0)
        de_sum += deltaE_ciede2000(lab_enh, lab_gt).mean()
        uiqm_sum += calc_uiqm(enh / 255.0)
        uciqe_sum += calc_uciqe(enh / 255.0)
        n += 1

    psnr = psnr_sum / n
    de = de_sum / n
    uiqm = uiqm_sum / n
    uciqe = uciqe_sum / n

    print(f'\n=== Results ({n} images) ===')
    print(f'PSNR:  {psnr:.4f} dB')
    if args.known_psnr:
        ok = abs(psnr - args.known_psnr) < 0.1
        print(f'  vs known {args.known_psnr}: {"OK" if ok else "MISMATCH"}')
    print(f'dE:    {de:.4f}')
    print(f'UIQM:  {uiqm:.4f}')
    print(f'UCIQE: {uciqe:.4f}')
    print(f'\n__RESULT__ {psnr:.4f} {de:.4f} {uiqm:.4f} {uciqe:.4f}')


if __name__ == '__main__':
    main()
