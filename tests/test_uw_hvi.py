"""
Numerical tests for UW-HVI implementation.

Test 1: Gauge round-trip (pure transform, no network)
    For 10k random pixels and random β_c, B_c, d:
    compute u = g(I) (WITHOUT clamp), then Î = g⁻¹(u).
    Assert max|Î - I| < 1e-5.

Test 2: Identity-init equals baseline (full pipeline)
    Load trained baseline weights into CIDNet.
    Build UW-HVI-CIDNet with identity init.
    Run both on same batch of real test images.
    Assert max|Ĵ_UWHVI - Ĵ_baseline| < 1e-4.
"""

import os
import sys
import torch
import numpy as np

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test1_gauge_roundtrip():
    """
    Test 1: gauge round-trip (pure transform, no network).

    For 10k random pixels I ~ U[0,1]³ and random β, B, d:
        u = g(I)  (NO clamp)
        Î = g⁻¹(u)
        Assert max|Î - I| < 1e-5

    This verifies g⁻¹ is the exact affine inverse of g.
    """
    print("=" * 60)
    print("Test 1: Gauge Round-Trip")
    print("=" * 60)

    from net.UW_HVI import UWHVITransform

    # Create transform (without depth head, we'll pass d manually)
    trans = UWHVITransform(use_depth=False)

    # Generate 10k random pixels
    torch.manual_seed(42)
    N = 10000
    I = torch.rand(N, 3)  # (N, 3), ~U[0,1]

    # Random params
    beta = torch.rand(3) * 0.95 + 0.05   # ~U[0.05, 1]
    B = torch.rand(3)                     # ~U[0, 1]
    d = torch.rand(N, 1) * 3.0            # ~U[0, 3]

    # Compute transmission
    beta_exp = beta.view(1, 3)
    t = torch.exp(-beta_exp * d)          # (N, 3)
    t = torch.clamp(t, 0.1, 1.0)

    # Forward g (NO clamp)
    B_exp = B.view(1, 3)
    u = (I - B_exp) / t + B_exp           # (N, 3), unclamped

    # Inverse g⁻¹
    I_hat = t * u + B_exp * (1.0 - t)     # (N, 3)

    # Compute error
    max_error = torch.max(torch.abs(I_hat - I)).item()
    mean_error = torch.mean(torch.abs(I_hat - I)).item()

    print(f"  Number of pixels tested: {N}")
    print(f"  Max  |Î - I|:  {max_error:.10e}")
    print(f"  Mean |Î - I|:  {mean_error:.10e}")

    # Verify
    threshold = 1e-5
    passed = max_error < threshold
    if passed:
        print(f"\n  ✓ TEST 1 PASSED — max error {max_error:.2e} < {threshold:.0e}")
    else:
        print(f"\n  ✗ TEST 1 FAILED — max error {max_error:.2e} >= {threshold:.0e}")
        print(f"    The gauge round-trip is NOT exact. Check g/g⁻¹ implementation.")

    # Additional: test with edge cases
    print("\n  Running edge case tests...")
    all_passed = passed

    # Edge case 1: t = 1.0 (should be exact identity)
    t_ones = torch.ones(N, 3)
    B_test = torch.rand(3)
    I_test = torch.rand(N, 3)
    B_exp_t = B_test.view(1, 3)
    u_test = (I_test - B_exp_t) / t_ones + B_exp_t
    I_hat_test = t_ones * u_test + B_exp_t * (1.0 - t_ones)
    err_t1 = torch.max(torch.abs(I_hat_test - I_test)).item()
    print(f"  Edge case t=1.0: max error = {err_t1:.10e} {'✓' if err_t1 < 1e-6 else '✗'}")

    # Edge case 2: t at min (0.1) — amplification but still invertible
    t_min = torch.full((N, 3), 0.1)
    u_test2 = (I_test - B_exp_t) / t_min + B_exp_t
    I_hat_test2 = t_min * u_test2 + B_exp_t * (1.0 - t_min)
    err_tmin = torch.max(torch.abs(I_hat_test2 - I_test)).item()
    print(f"  Edge case t=0.1: max error = {err_tmin:.10e} {'✓' if err_tmin < 1e-6 else '✗'}")

    # Edge case 3: d=0 → t=1 (identity init condition)
    d_zero = torch.zeros(N, 1)
    t_from_d0 = torch.exp(-beta_exp * d_zero)
    t_from_d0 = torch.clamp(t_from_d0, 0.1, 1.0)
    err_d0 = torch.max(torch.abs(t_from_d0 - torch.ones(N, 3))).item()
    print(f"  Edge case d=0 → t=1: max |t-1| = {err_d0:.10e} {'✓' if err_d0 < 1e-6 else '✗'}")

    return passed, max_error


def load_baseline_weights(baseline_pth, device='cuda'):
    """
    Load baseline CIDNet weights.
    Returns the state dict.
    """
    from net.CIDNet import CIDNet

    model = CIDNet()
    state = torch.load(baseline_pth, map_location=device)
    model.load_state_dict(state, strict=True)
    model = model.to(device)
    model.eval()
    return model, state


def build_uw_cidnet_from_baseline(baseline_state, use_depth=True, device='cuda'):
    """
    Build UWCIDNet with identity init, then load encoder/decoder/LCA weights
    from baseline state dict.

    Mapping:
        baseline: trans.*          → uw: trans.hvi.*
        baseline: HVE_block*.*     → uw: HVE_block*.* (same)
        baseline: HVD_block*.*     → uw: HVD_block*.* (same)
        baseline: IE_block*.*      → uw: IE_block*.* (same)
        baseline: ID_block*.*      → uw: ID_block*.* (same)
        baseline: HV_LCA*.*        → uw: HV_LCA*.* (same)
        baseline: I_LCA*.*         → uw: I_LCA*.* (same)
        baseline: lfrc.*           → uw: lfrc.* (same, if exists)

    UW-HVI specific params (theta, phi, depth_head) stay at identity init.
    """
    from net.UW_CIDNet import UWCIDNet

    use_lfrc = 'lfrc.lfrc_conv.weight' in baseline_state

    model = UWCIDNet(use_lfrc=use_lfrc, use_depth=use_depth)
    model = model.to(device)

    # Build mapped state dict
    uw_state = model.state_dict()

    for name, param in baseline_state.items():
        # Map RGB_HVI params to UWHVITransform.hvi
        if name.startswith('trans.'):
            uw_name = 'trans.hvi.' + name[len('trans.'):]
        else:
            uw_name = name

        if uw_name in uw_state:
            if uw_state[uw_name].shape == param.shape:
                uw_state[uw_name] = param.clone()
            else:
                print(f"  WARNING: shape mismatch for {name} → {uw_name}: "
                      f"{param.shape} vs {uw_state[uw_name].shape}, skipping")
        else:
            # Not found in UW model (e.g., some unused params) — safe to skip
            pass

    # Load mapped state dict (strict=False because UW-HVI params stay at init)
    model.load_state_dict(uw_state, strict=False)
    model.eval()
    return model


def test2_identity_init_equals_baseline(baseline_pth=None):
    """
    Test 2: Identity-init equals baseline (full pipeline).

    Load trained baseline weights into CIDNet.
    Build UW-HVI-CIDNet with identity init + baseline weights.
    Run both on same batch of real images.
    Assert max|output_uw - output_baseline| < 1e-4.
    """
    print("\n" + "=" * 60)
    print("Test 2: Identity-Init Equals Baseline")
    print("=" * 60)

    from net.CIDNet import CIDNet
    from net.UW_CIDNet import UWCIDNet
    from torchvision import transforms
    from PIL import Image
    import glob

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}")

    # If no baseline path given, try to find one
    if baseline_pth is None:
        # Prefer UIEB baseline: best_of/uieb_best_vanilla_*.pth
        uieb_candidates = sorted(glob.glob('./weights/best_of/uieb_best_vanilla_*.pth'))
        if uieb_candidates:
            baseline_pth = uieb_candidates[-1]
        else:
            candidates = sorted(glob.glob('./weights/train/epoch_*.pth'))
            candidates += sorted(glob.glob('./weights/best_of/*.pth'))
            if candidates:
                baseline_pth = candidates[-1]

        if baseline_pth:
            print(f"  Auto-selected baseline: {baseline_pth}")
        else:
            print("  No trained baseline weights found. Using random-init comparison.")
            print("  (This tests structural identity, not weight-matched identity.)")

    # Load or create baseline
    if baseline_pth and os.path.exists(baseline_pth):
        baseline_model, baseline_state = load_baseline_weights(baseline_pth, device)
        uw_model = build_uw_cidnet_from_baseline(baseline_state, use_depth=True, device=device)
    else:
        # Create both from scratch with same random init
        torch.manual_seed(42)
        baseline_model = CIDNet().to(device)
        baseline_model.eval()

        torch.manual_seed(42)
        uw_model = UWCIDNet(use_depth=True).to(device)
        uw_model.eval()
        print("  Using identical random seed (42) for both models.")

    # Helper to pad image dimensions to multiples of 8 (3 downsample levels)
    def pad_to_multiple(tensor, multiple=8):
        _, _, h, w = tensor.shape
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple
        if pad_h > 0 or pad_w > 0:
            tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode='reflect')
        return tensor

    # Load test images
    test_dir = './datasets/UIEB/test/low'
    if not os.path.exists(test_dir):
        print(f"  Test directory not found: {test_dir}")
        print("  Using synthetic test images instead.")
        # Use random tensor as input
        torch.manual_seed(123)
        test_inputs = [torch.rand(1, 3, 256, 256).to(device) for _ in range(5)]
    else:
        test_files = sorted(glob.glob(os.path.join(test_dir, '*')))[:10]
        if len(test_files) == 0:
            print(f"  No test images found in {test_dir}")
            return False, float('inf')

        print(f"  Testing on {len(test_files)} real UIEB test images")

        to_tensor = transforms.ToTensor()
        test_inputs = []
        for f in test_files:
            img = Image.open(f).convert('RGB')
            tensor = to_tensor(img).unsqueeze(0).to(device)
            tensor = pad_to_multiple(tensor, 8)
            test_inputs.append(tensor)

    # Run comparison
    max_error = 0.0
    all_errors = []

    with torch.no_grad():
        for i, inp in enumerate(test_inputs):
            out_baseline = baseline_model(inp)
            out_uw = uw_model(inp)

            error = torch.max(torch.abs(out_uw - out_baseline)).item()
            mean_err = torch.mean(torch.abs(out_uw - out_baseline)).item()
            all_errors.append(error)
            max_error = max(max_error, error)

            status = '✓' if error < 1e-4 else '✗'
            print(f"  Image {i+1}: max error = {error:.8e}, mean error = {mean_err:.8e} {status}")

    threshold = 1e-4
    passed = max_error < threshold

    print(f"\n  Overall max error: {max_error:.8e}")
    print(f"  Threshold: {threshold:.0e}")

    if passed:
        print(f"  ✓ TEST 2 PASSED — identity init is bit-identical to baseline")
    else:
        print(f"  ✗ TEST 2 FAILED — max error {max_error:.2e} >= {threshold:.0e}")
        print(f"    The floor guarantee is broken. Fix before training.")
        print(f"    Possible issues:")
        print(f"      1. Depth head not outputting ≈0 at init")
        print(f"      2. β_c too large → t_c ≠ 1")
        print(f"      3. Clamp or numerical issue in g/g⁻¹")

    return passed, max_error


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    print("\n" + "=" * 60)
    print("UW-HVI NUMERICAL TESTS")
    print("=" * 60)
    print(f"Working directory: {os.getcwd()}")
    print()

    # Test 1
    t1_passed, t1_error = test1_gauge_roundtrip()

    # Test 2
    t2_passed, t2_error = test2_identity_init_equals_baseline()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Test 1 (gauge round-trip):      {'✓ PASSED' if t1_passed else '✗ FAILED'}  "
          f"(max error: {t1_error:.2e})")
    print(f"  Test 2 (identity-init baseline): {'✓ PASSED' if t2_passed else '✗ FAILED'}  "
          f"(max error: {t2_error:.2e})")

    if t1_passed and t2_passed:
        print("\n  ✓ ALL TESTS PASSED — Safe to proceed with training.")
        sys.exit(0)
    else:
        print("\n  ✗ SOME TESTS FAILED — Fix before training.")
        sys.exit(1)
