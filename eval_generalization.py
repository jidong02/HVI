#!/usr/bin/env python3
"""Evaluate arbitrary images: inference + UIQM/UCIQE.
Usage: python eval_generalization.py --weight_path <pth> --input_dir <dir> [--output_dir <dir>]"""
import os, sys, glob, argparse
import torch, torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from net.CIDNet import CIDNet
from metrics.no_reference import calc_uiqm, calc_uciqe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', type=str, required=True)
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = args.device if args.device == 'cuda' and torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    model = CIDNet().to(device)
    sd = torch.load(args.weight_path, map_location=device)
    model.load_state_dict(sd, strict=False)
    model.eval()
    print(f'Loaded: {args.weight_path}')

    files = sorted(glob.glob(os.path.join(args.input_dir, '*')))
    files = [f for f in files if os.path.isfile(f) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    if not files:
        print(f'No images found in {args.input_dir}')
        return

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    factor = 8
    uiqm_list, uciqe_list = [], []

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
        save_path = os.path.join(args.output_dir, name) if args.output_dir else f'/tmp/eval_gen_tmp_{name}'
        out_img.save(save_path)

        enh = np.array(Image.open(save_path).convert('RGB')) / 255.0
        uiqm_list.append(calc_uiqm(enh))
        uciqe_list.append(calc_uciqe(enh))

        if not args.output_dir:
            os.remove(save_path)

    uiqm_arr = np.array(uiqm_list)
    uciqe_arr = np.array(uciqe_list)
    print(f'\n=== Results ({len(files)} images) ===')
    print(f'UIQM:  {uiqm_arr.mean():.4f} ± {uiqm_arr.std():.4f}')
    print(f'UCIQE: {uciqe_arr.mean():.4f} ± {uciqe_arr.std():.4f}')


if __name__ == '__main__':
    main()
