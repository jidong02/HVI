#!/usr/bin/env python3
"""Visualize HVI-CIDNet training logs with matplotlib."""
import re, sys, argparse
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # headless-safe

def parse_log(path):
    epochs, losses = [], []
    val_epochs, psnrs, ssims, lpipss = [], [], [], []

    with open(path) as f:
        for line in f:
            # Training loss: ===> Epoch[N]: Loss: X || Learning rate: lr=Y.
            m = re.match(r"===>\s*Epoch\[(\d+)\]:\s*Loss:\s*([\d.]+)", line)
            if m:
                epochs.append(int(m.group(1)))
                losses.append(float(m.group(2)))

            # Validation metrics
            m = re.match(r"===>\s*Avg\.PSNR:\s*([\d.]+)\s*dB", line)
            if m:
                psnrs.append(float(m.group(1)))
            m = re.match(r"===>\s*Avg\.SSIM:\s*([\d.]+)", line)
            if m:
                ssims.append(float(m.group(1)))
            m = re.match(r"===>\s*Avg\.LPIPS:\s*([\d.]+)", line)
            if m:
                lpipss.append(float(m.group(1)))

    # val_epochs are every snapshots epoch. Find them by looking at PSNR occurrence positions.
    # Simpler: val occurs at multiples of snapshots. Let's infer from loss data.
    # Actually val is printed after checkpoint(). Let's just assign val_epochs from loss epochs at the
    # same indices — the val block appears right after the checkpoint epoch's loss line.
    # We scan the file directly.

    return epochs, losses, psnrs, ssims, lpipss


def parse_log_v2(path):
    """Better parser: track val epoch from surrounding loss lines."""
    epochs, losses = [], []
    val_epochs, psnrs, ssims, lpipss = [], [], [], []
    current_epoch = None

    with open(path) as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        # Training loss
        m = re.match(r"===>\s*Epoch\[(\d+)\]:\s*Loss:\s*([\d.]+)", line)
        if m:
            current_epoch = int(m.group(1))
            epochs.append(current_epoch)
            losses.append(float(m.group(2)))

        # Validation metrics — use the most recent epoch (the checkpoint epoch)
        m = re.match(r"===>\s*Avg\.PSNR:\s*([\d.]+)\s*dB", line)
        if m and current_epoch is not None:
            psnrs.append(float(m.group(1)))
            val_epochs.append(current_epoch)

        m = re.match(r"===>\s*Avg\.SSIM:\s*([\d.]+)", line)
        if m and current_epoch is not None:
            ssims.append(float(m.group(1)))

        m = re.match(r"===>\s*Avg\.LPIPS:\s*([\d.]+)", line)
        if m and current_epoch is not None:
            lpipss.append(float(m.group(1)))

    return epochs, losses, val_epochs, psnrs, ssims, lpipss


def plot(log_path, out_path):
    epochs, losses, val_epochs, psnrs, ssims, lpipss = parse_log_v2(log_path)

    if not epochs:
        print("No training data found in log!")
        sys.exit(1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Training Log: {log_path.split("/")[-1]}', fontsize=14, fontweight='bold')

    # Loss curve
    ax = axes[0, 0]
    ax.plot(epochs, losses, 'b-', linewidth=1, alpha=0.7, label='Train Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # PSNR
    ax = axes[0, 1]
    ax.plot(val_epochs, psnrs, 'g-o', markersize=5, linewidth=1.5, label='PSNR (dB)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('PSNR (dB)')
    ax.set_title('Validation PSNR')
    ax.legend()
    ax.grid(True, alpha=0.3)
    # Annotate best
    if psnrs:
        best_idx = max(range(len(psnrs)), key=lambda i: psnrs[i])
        ax.annotate(f'{psnrs[best_idx]:.2f}',
                    (val_epochs[best_idx], psnrs[best_idx]),
                    textcoords="offset points", xytext=(0, 10),
                    fontsize=9, color='green', fontweight='bold')

    # SSIM
    ax = axes[1, 0]
    ax.plot(val_epochs, ssims, 'c-o', markersize=5, linewidth=1.5, label='SSIM')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('SSIM')
    ax.set_title('Validation SSIM')
    ax.legend()
    ax.grid(True, alpha=0.3)
    if ssims:
        best_idx = max(range(len(ssims)), key=lambda i: ssims[i])
        ax.annotate(f'{ssims[best_idx]:.4f}',
                    (val_epochs[best_idx], ssims[best_idx]),
                    textcoords="offset points", xytext=(0, 10),
                    fontsize=9, color='teal', fontweight='bold')

    # LPIPS
    ax = axes[1, 1]
    ax.plot(val_epochs, lpipss, 'r-o', markersize=5, linewidth=1.5, label='LPIPS')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('LPIPS')
    ax.set_title('Validation LPIPS (lower is better)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    if lpipss:
        best_idx = min(range(len(lpipss)), key=lambda i: lpipss[i])
        ax.annotate(f'{lpipss[best_idx]:.4f}',
                    (val_epochs[best_idx], lpipss[best_idx]),
                    textcoords="offset points", xytext=(0, 10),
                    fontsize=9, color='red', fontweight='bold')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {out_path}')

    # Print summary
    print(f'\n=== Summary ({len(epochs)} epochs, {len(val_epochs)} checkpoints) ===')
    print(f'Train Loss: {losses[0]:.4f} → {losses[-1]:.4f}')
    if psnrs:
        best_psnr = max(zip(val_epochs, psnrs), key=lambda x: x[1])
        best_ssim = max(zip(val_epochs, ssims), key=lambda x: x[1])
        best_lpips = min(zip(val_epochs, lpipss), key=lambda x: x[1])
        print(f'Best PSNR:  {best_psnr[1]:.4f} dB @ epoch {best_psnr[0]}')
        print(f'Best SSIM:  {best_ssim[1]:.4f} @ epoch {best_ssim[0]}')
        print(f'Best LPIPS: {best_lpips[1]:.4f} @ epoch {best_lpips[0]}')
        last = max(val_epochs)
        print(f'Final PSNR: {psnrs[-1]:.4f} dB')
        print(f'Final SSIM: {ssims[-1]:.4f}')
        print(f'Final LPIPS: {lpipss[-1]:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize HVI-CIDNet training log')
    parser.add_argument('log', nargs='?', default='training_logs/log_b1_red.txt',
                        help='Path to training log file')
    parser.add_argument('-o', '--output', default=None,
                        help='Output image path (default: <log>_plot.png)')
    args = parser.parse_args()

    out = args.output or args.log.replace('.txt', '_plot.png')
    plot(args.log, out)
