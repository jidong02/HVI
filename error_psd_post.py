"""
error_psd_post.py
=================
Post-LFRC error-PSD analysis in HVI space.

Goal: close the diagnosis loop. The original error-PSD said ~90% of prediction-
error energy is low-frequency and ~69% achromatic. LFRC was designed to attack
exactly that band. This module measures, over the whole test set, whether the
error PSD actually changes after LFRC -- BEFORE vs AFTER, achromatic (I) vs
chromatic (HV) -- and how much LF energy is removed.

Everything is computed in HVI space, i.e. the same space LFRC operates in
(after the backbone, before PHVIT). For each test sample you feed:

    pre  : (i, hv) from the backbone output      <- LFRC input
    post : (i, hv) after LFRC correction         <- LFRC output
    gt   : (i, hv) of the ground truth           <- RGB->HVI(gt) via your transform

Radial PSD uses NORMALISED frequency (0 = DC, 1 = Nyquist) so that UIEB's
variable-resolution test images can be pooled into the same bins.

Usage (inside the eval loop)
----------------------------
    an = ErrorPSD(nbins=128)
    for sample in loader:
        i_pre, hv_pre   = backbone(x)            # detach to numpy/cpu
        i_post, hv_post = lfrc(i_pre, hv_pre, x) # detach to numpy/cpu
        i_gt, hv_gt     = rgb2hvi(gt)            # detach to numpy/cpu
        an.add(i_pre, hv_pre, i_post, hv_post, i_gt, hv_gt)
    an.report(lf_cut=0.25, hf_start=0.5)         # MATCH your original LF cutoff
    an.plot("psd_pre_post.png", lf_cut=0.25)

Channel conventions accepted by add():
    i  : (H,W) or (1,H,W) or (H,W,1)
    hv : (2,H,W) or (H,W,2)              (H and V stacked; both count as chromatic)
Values in whatever range your HVI transform produces -- only the ERROR is used,
so absolute scaling is irrelevant as long as pre/post/gt share it.
"""

import numpy as np


def _to_2d_list(arr):
    """Normalise an input into a list of 2D float arrays (one per channel)."""
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 2:                      # (H,W)
        return [a]
    if a.ndim == 3:
        # find the channel axis: the smallest dim of size 1,2,3 is treated as channels
        if a.shape[0] in (1, 2, 3) and a.shape[0] <= a.shape[-1]:
            return [a[c] for c in range(a.shape[0])]      # (C,H,W)
        if a.shape[-1] in (1, 2, 3):
            return [a[..., c] for c in range(a.shape[-1])]  # (H,W,C)
    raise ValueError(f"Unexpected array shape {a.shape}; expected (H,W) or (C,H,W)/(H,W,C) with C in 1..3")


def _radial_energy(err2d, nbins):
    """Total spectral power per normalised-frequency bin (length nbins).

    By Parseval, sum over bins == sum(err2d**2) (within FFT normalisation),
    so integrating these bins gives spatial-domain error energy.
    """
    F = np.fft.fftshift(np.fft.fft2(err2d))
    P = (np.abs(F) ** 2) / err2d.size      # Parseval-consistent normalisation
    h, w = P.shape
    cy, cx = h / 2.0, w / 2.0
    y, x = np.indices((h, w))
    # normalised radius: 1.0 at Nyquist along an axis; corners clipped to 1.0
    r = np.sqrt(((x - cx) / (w / 2.0)) ** 2 + ((y - cy) / (h / 2.0)) ** 2)
    r = np.clip(r, 0.0, 1.0)
    idx = np.minimum((r * nbins).astype(int), nbins - 1)
    energy = np.bincount(idx.ravel(), weights=P.ravel(), minlength=nbins)
    count = np.bincount(idx.ravel(), minlength=nbins)
    return energy.astype(np.float64), count.astype(np.float64)


class ErrorPSD:
    def __init__(self, nbins=128):
        self.nbins = nbins
        # energy accumulators (summed over images), split by stage x component
        self.E = {("pre", "ach"): np.zeros(nbins), ("pre", "chr"): np.zeros(nbins),
                  ("post", "ach"): np.zeros(nbins), ("post", "chr"): np.zeros(nbins)}
        self.C = np.zeros(nbins)   # bin counts (shared geometry), for radial averaging
        self.n = 0

    def _accum(self, stage, i_err, hv_err):
        # achromatic = I channel
        for ch in _to_2d_list(i_err):
            e, c = _radial_energy(ch, self.nbins)
            self.E[(stage, "ach")] += e
            self.C += c
        # chromatic = H and V channels
        for ch in _to_2d_list(hv_err):
            e, c = _radial_energy(ch, self.nbins)
            self.E[(stage, "chr")] += e
            self.C += c

    def add(self, i_pre, hv_pre, i_post, hv_post, i_gt, hv_gt):
        i_gt_l = _to_2d_list(i_gt)
        i_pre_l, i_post_l = _to_2d_list(i_pre), _to_2d_list(i_post)
        hv_gt_l = _to_2d_list(hv_gt)
        hv_pre_l, hv_post_l = _to_2d_list(hv_pre), _to_2d_list(hv_post)

        def diff(a_l, b_l):
            return [a - b for a, b in zip(a_l, b_l)]

        # stack back so _accum gets channel-lists with matching geometry
        self._accum("pre", np.stack(diff(i_pre_l, i_gt_l)), np.stack(diff(hv_pre_l, hv_gt_l)))
        self._accum("post", np.stack(diff(i_post_l, i_gt_l)), np.stack(diff(hv_post_l, hv_gt_l)))
        self.n += 1

    # ---- reporting -------------------------------------------------------
    def _totals(self, stage):
        ach = self.E[(stage, "ach")]
        chr_ = self.E[(stage, "chr")]
        return ach, chr_, ach + chr_

    def report(self, lf_cut=0.25, hf_start=0.5):
        """Print pre/post energy decomposition. lf_cut / hf_start are fractions
        of Nyquist; lf_cut MUST match the cutoff your original error_psd.py used."""
        lf_b = int(round(lf_cut * self.nbins))
        hf_b = int(round(hf_start * self.nbins))

        def frac(arr, lo, hi):
            tot = arr.sum()
            return 0.0 if tot == 0 else arr[lo:hi].sum() / tot

        rows = []
        for stage in ("pre", "post"):
            ach, chr_, tot = self._totals(stage)
            rows.append(dict(
                stage=stage,
                total=tot.sum(),
                lf_frac=frac(tot, 0, lf_b),
                hf_frac=frac(tot, hf_b, self.nbins),
                ach_frac=ach.sum() / tot.sum(),
                chr_frac=chr_.sum() / tot.sum(),
                lf_energy=tot[:lf_b].sum(),
            ))

        pre, post = rows
        print(f"# images: {self.n} | nbins: {self.nbins} | "
              f"LF cutoff: <{lf_cut:.3f} Nyq | HF band: >{hf_start:.3f} Nyq\n")
        hdr = f"{'':6s}{'total E':>12s}{'LF frac':>9s}{'HF frac':>9s}{'achrom':>9s}{'chrom':>9s}{'LF energy':>13s}"
        print(hdr)
        for r in rows:
            print(f"{r['stage']:6s}{r['total']:12.4g}{r['lf_frac']:9.3f}"
                  f"{r['hf_frac']:9.3f}{r['ach_frac']:9.3f}{r['chr_frac']:9.3f}{r['lf_energy']:13.4g}")

        d_tot = (post['total'] - pre['total']) / pre['total'] * 100
        d_lf = (post['lf_energy'] - pre['lf_energy']) / pre['lf_energy'] * 100
        print(f"\nTotal error energy: {d_tot:+.1f}%   (should track PSNR gain; "
              f"negative = error reduced)")
        print(f"LF  error energy:   {d_lf:+.1f}%   "
              f"(the keystone number: LFRC is supposed to drive this down)")
        print(f"LF fraction:        {pre['lf_frac']:.3f} -> {post['lf_frac']:.3f}   "
              f"(shrinks if the removed energy was concentrated in LF)")
        return dict(pre=pre, post=post)

    def plot(self, path, lf_cut=0.25):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        f = (np.arange(self.nbins) + 0.5) / self.nbins  # bin-centre normalised freq
        radial = {k: np.divide(v, np.maximum(self.C, 1)) for k, v in self.E.items()}

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
        for ax, comp, title in ((axes[0], "ach", "Achromatic (I)"),
                                (axes[1], "chr", "Chromatic (HV)")):
            ax.loglog(f, radial[("pre", comp)] + 1e-20, label="pre-LFRC (backbone)", lw=1.8)
            ax.loglog(f, radial[("post", comp)] + 1e-20, label="post-LFRC", lw=1.8)
            ax.axvspan(f[0], lf_cut, alpha=0.10, color="tab:blue")
            ax.text(f[0] * 1.1, ax.get_ylim()[1], "LF band", va="top", fontsize=8, alpha=0.7)
            ax.set_title(title)
            ax.set_xlabel("normalised spatial frequency (1 = Nyquist)")
            ax.grid(True, which="both", alpha=0.25)
            ax.legend(fontsize=8)
        axes[0].set_ylabel("radially-averaged error PSD")
        fig.suptitle("Error PSD before vs after LFRC (HVI space)")
        fig.tight_layout()
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"saved figure -> {path}")


# ===========================================================================
# SELF-TEST (synthetic) -- validates that the analysis detects exactly the
# expected behaviour. Run: python3 error_psd_post.py
# It fabricates an error field that is LF-dominated (like the real diagnosis),
# then simulates an "ideal LFRC" that low-pass-subtracts the LF part, and
# checks that (a) pre LF-fraction is high, (b) LF energy drops sharply post.
# ===========================================================================
def _make_lf_dominated_error(h, w, rng, lf_scale=1.0, hf_scale=0.04):
    """Smooth (LF) field + faint white (broadband) noise."""
    # LF: low-frequency random field via blurred noise
    base = rng.standard_normal((h, w))
    Fb = np.fft.fftshift(np.fft.fft2(base))
    yy, xx = np.indices((h, w))
    r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    lp = np.exp(-(r / 0.06) ** 2)              # keep only very low freqs
    lf = np.real(np.fft.ifft2(np.fft.ifftshift(Fb * lp)))
    lf *= lf_scale / (lf.std() + 1e-9)
    hf = hf_scale * rng.standard_normal((h, w))  # broadband (incl. HF) noise
    return lf + hf, lf  # return field and its LF component (what an ideal LFRC removes)


def _self_test():
    rng = np.random.default_rng(0)
    an = ErrorPSD(nbins=128)
    sizes = [(256, 256), (200, 300), (180, 240)]  # variable resolution, like UIEB
    for _ in range(12):
        h, w = sizes[rng.integers(len(sizes))]
        # I (achromatic): strong LF error. HV (chromatic): weaker LF error.
        ei, lf_i = _make_lf_dominated_error(h, w, rng, lf_scale=1.0)
        eh, lf_h = _make_lf_dominated_error(h, w, rng, lf_scale=0.35)
        ev, lf_v = _make_lf_dominated_error(h, w, rng, lf_scale=0.35)

        # "GT" is zero in error-space; pre = error, post = error minus 80% of its LF part
        # (an imperfect but real LFRC that mostly removes LF, leaves HF untouched)
        i_pre, hv_pre = ei, np.stack([eh, ev])
        i_post = ei - 0.8 * lf_i
        hv_post = np.stack([eh - 0.8 * lf_h, ev - 0.8 * lf_v])
        z_i, z_hv = np.zeros((h, w)), np.zeros((2, h, w))
        an.add(i_pre, hv_pre, i_post, hv_post, z_i, z_hv)

    print("=" * 78)
    print("SELF-TEST (synthetic LF-dominated error + ideal LF-removing LFRC)")
    print("=" * 78)
    res = an.report(lf_cut=0.25, hf_start=0.5)
    an.plot("/home/claude/selftest_psd.png", lf_cut=0.25)

    pre, post = res["pre"], res["post"]
    # Assertions: the analysis must reflect the construction.
    assert pre["lf_frac"] > 0.85, f"expected LF-dominated pre, got {pre['lf_frac']:.3f}"
    assert pre["hf_frac"] < 0.05, f"expected tiny HF pre, got {pre['hf_frac']:.3f}"
    assert post["lf_energy"] < 0.25 * pre["lf_energy"], "LFRC should slash LF energy"
    assert post["total"] < pre["total"], "total error energy should drop"
    assert post["lf_frac"] < pre["lf_frac"], "LF fraction should shrink after LF removal"
    print("\nALL SELF-TEST ASSERTIONS PASSED  ✓")
    print("(pre is LF-dominated; LFRC removes LF; total + LF energy drop; "
          "LF fraction shrinks -- exactly the closed-loop signature you want on real data.)")


if __name__ == "__main__":
    _self_test()