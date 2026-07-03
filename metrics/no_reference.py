"""No-reference underwater image quality metrics: UIQM and UCIQE.

UIQM: 0.0282·UICM + 0.2953·UISM + 3.5753·UIConM  (Panetta standard)
UCIQE: 0.4680·σc + 0.2745·con_l + 0.2576·μs       (Yang 2015, skimage Lab scale)
"""
import numpy as np
from skimage.color import rgb2lab
from skimage.filters import sobel


def _uicm(img_rgb):
    """Underwater Image Colorfulness Measure. Input [0,1] RGB."""
    rg = img_rgb[..., 0] - img_rgb[..., 1]
    yb = 0.5 * (img_rgb[..., 0] + img_rgb[..., 1]) - img_rgb[..., 2]
    mu_rg = rg.mean()
    mu_yb = yb.mean()
    sigma_rg = rg.std()
    sigma_yb = yb.std()
    return np.sqrt(sigma_rg**2 + sigma_yb**2) + 0.3 * np.sqrt(mu_rg**2 + mu_yb**2)


def _uism_emap(img_rgb):
    """Edge map for UISM using Sobel on R, G, B channels separately."""
    R = img_rgb[..., 0]
    G = img_rgb[..., 1]
    B = img_rgb[..., 2]
    eR = sobel(R)
    eG = sobel(G)
    eB = sobel(B)
    emap = np.sqrt(eR**2 + eG**2 + eB**2)
    return emap


def _uism(img_rgb):
    """Underwater Image Sharpness Measure."""
    emap = _uism_emap(img_rgb)
    # Divide into 16×16 blocks (or roughly), compute EME in each block
    h, w = emap.shape
    bh, bw = max(1, h // 16), max(1, w // 16)
    nh, nw = h // bh, w // bw
    sum_eme = 0.0
    n_blocks = 0
    for i in range(nh):
        for j in range(nw):
            block = emap[i * bh : (i + 1) * bh, j * bw : (j + 1) * bw]
            mx = block.max()
            mn = block.min()
            if mx > 0 and mn > 0:
                sum_eme += 2.0 * np.log((mx + 1e-9) / (mn + 1e-9)) * (mx / (mx + mn + 1e-9))
                n_blocks += 1
    return sum_eme / max(n_blocks, 1)


def _uiconm(img_rgb):
    """Underwater Image Contrast Measure."""
    gray = 0.299 * img_rgb[..., 0] + 0.587 * img_rgb[..., 1] + 0.114 * img_rgb[..., 2]
    h, w = gray.shape
    bh, bw = max(1, h // 16), max(1, w // 16)
    nh, nw = h // bh, w // bw
    log_ame = []
    for i in range(nh):
        for j in range(nw):
            block = gray[i * bh : (i + 1) * bh, j * bw : (j + 1) * bw]
            mx = block.max()
            mn = block.min()
            if mx > 0 and mn > 0:
                log_ame.append(np.log((mx - mn + 1e-9) / (mx + mn + 1e-9)))
    log_ame_mean = np.mean(log_ame) if log_ame else 0.0
    return np.exp(log_ame_mean) if log_ame_mean != 0 else 0.0


def calc_uiqm(img_rgb):
    """Compute UIQM for a single [0,1] RGB image (H×W×3)."""
    uicm = _uicm(img_rgb)
    uism = _uism(img_rgb)
    uiconm = _uiconm(img_rgb)
    return 0.0282 * uicm + 0.2953 * uism + 3.5753 * uiconm


def calc_uciqe(img_rgb):
    """Compute UCIQE for a single [0,1] RGB image (H×W×3).
    Uses cv2 RGB→LAB, then L/a/b normalized to [0,~1].
    UCIQE = 0.4680·σc + 0.2745·con_l + 0.2576·μs
    Reference range: ~0.3-0.7 for typical underwater images."""
    import cv2
    rgb_uint8 = (np.clip(img_rgb, 0, 1) * 255).astype(np.uint8)
    lab = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB).astype(np.float64)
    L = lab[..., 0] / 255.0
    a = lab[..., 1] / 255.0
    b = lab[..., 2] / 255.0

    chroma = np.sqrt(a**2 + b**2)
    sigma_c = np.std(chroma)

    # Saturation: chroma / sqrt(chroma^2 + L^2)
    sat = chroma / np.sqrt(chroma**2 + L**2 + 1e-8)
    mu_s = sat.mean()

    # Luminance contrast: trimmed 1% top vs 1% bottom
    Lf = np.sort(L.flatten())
    k = max(1, int(round(0.01 * Lf.size)))
    con_l = Lf[-k:].mean() - Lf[:k].mean()

    return 0.4680 * sigma_c + 0.2745 * con_l + 0.2576 * mu_s
