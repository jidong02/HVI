"""Run post-LFRC error PSD on UIEB test, using LFRC v2 checkpoint."""
import torch, os, glob, numpy as np
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from error_psd_post import ErrorPSD

CR = '/root/autodl-tmp/HVI-CIDNet'

# Temporarily add LFRC to CIDNet
from net.CIDNet import CIDNet
# Need to restore LFRC support — use eval_colors approach: load strict=False
# CIDNet needs use_lfrc=True to have lfrc module; but current CIDNet doesn't have it
# Workaround: temporarily modify the import chain

# Better approach: write a custom forward that captures intermediates
import sys
sys.path.insert(0, CR)

# Monkey-patch: add LFRC to the current vanilla CIDNet
from net.lf_corrector import ILowFreqCorrector

# Load model
model = CIDNet().cuda()
# Add lfrc manually
model.use_lfrc = True
model.lfrc = ILowFreqCorrector().cuda()

# Load v2 checkpoint
sd = torch.load('weights/lfrcv2_lfrc/epoch_50.pth', map_location='cuda')
# Filter lfrc keys only
lfrc_sd = {k.replace('lfrc.', ''): v for k, v in sd.items() if k.startswith('lfrc.')}
# Load backbone
model.load_state_dict({k: v for k, v in sd.items() if not k.startswith('lfrc.')}, strict=False)
# Load lfrc separately
model.lfrc.load_state_dict(lfrc_sd, strict=False)
model.eval()

# GT HVI transform
from net.HVI_transform import RGB_HVI
hvi_trans = RGB_HVI().cuda()

# Run inference on UIEB test, capturing pre/post HVI
low_files = sorted(glob.glob(f'{CR}/datasets/UIEB/test/low/*'))
gt_dir = f'{CR}/datasets/UIEB/test/high/'
factor = 8

an = ErrorPSD(nbins=128)

for f in tqdm(low_files, desc='analyze'):
    name = os.path.basename(f)
    gt_f = os.path.join(gt_dir, name)
    if not os.path.exists(gt_f):
        continue

    # Low input + GT
    img = Image.open(f).convert('RGB')
    gt_img = Image.open(gt_f).convert('RGB')

    x = transforms.ToTensor()(img).unsqueeze(0).cuda()
    gt = transforms.ToTensor()(gt_img).unsqueeze(0).cuda()

    _, _, h, w = x.shape
    H = ((h + factor) // factor) * factor
    W = ((w + factor) // factor) * factor
    if H > h or W > w:
        x = F.pad(x, (0, W - w, 0, H - h), 'reflect')

    # Run backbone + capture pre-LFRC tensors via hook
    pre_i, pre_hv = [], []
    def hook_pre(module, input, output):
        pre_i.append(output[0].detach())  # i_dec0
        pre_hv.append(output[1].detach())  # hv_0
    # Actual: need to intercept hv_0 and i_dec0 before cat
    # Use a different approach — run backbone step by step through monkey-patched forward
    
    # Let's just run the full CIDNet forward with LFRC and another without
    with torch.inference_mode():
        # Full forward with LFRC (LFRC weights loaded)
        output_full = model(x)[:, :, :h, :w].clamp(0, 1)
        # Vanilla forward (disable LFRC)
        model.use_lfrc = False
        output_vanilla = model(x)[:, :, :h, :w].clamp(0, 1)
        model.use_lfrc = True

    # GT HVI
    with torch.inference_mode():
        gt_hvi = hvi_trans.HVIT(gt)

    # The pre/post HVI approach: pre = vanilla output (no LFRC), post = LFRC output
    # Use HVIT of outputs as approximation
    pre_hvi_t = hvi_trans.HVIT(output_vanilla)
    post_hvi_t = hvi_trans.HVIT(output_full)

    i_pre = pre_hvi_t[:, 2:3].squeeze(0).cpu().numpy()       # (1, H, W)
    hv_pre = pre_hvi_t[:, :2].squeeze(0).cpu().numpy()        # (2, H, W)
    i_post = post_hvi_t[:, 2:3].squeeze(0).cpu().numpy()
    hv_post = post_hvi_t[:, :2].squeeze(0).cpu().numpy()
    i_gt = gt_hvi[:, 2:3].squeeze(0).cpu().numpy()
    hv_gt = gt_hvi[:, :2].squeeze(0).cpu().numpy()

    an.add(i_pre, hv_pre, i_post, hv_post, i_gt, hv_gt)

res = an.report(lf_cut=0.250, hf_start=0.5)
an.plot("psd_pre_post.png", lf_cut=0.250)
print("\nDone. See psd_pre_post.png")
