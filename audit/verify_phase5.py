"""Phase 5 verification: losses (master §8). NUMERIC, not structural.

The required checks, and the reason each is a hard failure:
  * dCR and generic bands PIXEL-IDENTICAL -- otherwise a C-vs-B' difference could come from
    band geometry and P5 is confounded.
  * dCR weights corners > smooth regions on a synthetic shape -- otherwise the "curvature"
    weighting is not actually curvature-aware and C is just a rescaled B'.
  * calibration CHANGES beta and EQUALISES gradient norms -- a no-op calibration would leave
    B' and C differing in boundary-term strength as well as placement.
  * gradients finite and non-NaN, edge cases finite.
"""
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

failures = []


def check(label, cond):
    ok = bool(cond)
    print(f"[{'PASS' if ok else 'FAIL'}] {label}")
    if not ok:
        failures.append(label)
    return ok


print("=" * 78)
print("PHASE 5 VERIFICATION -- losses (master §8)")
print("=" * 78)

from src.losses.boundary_band import (extract_boundary_band, band_mean,  # noqa: E402
                                      assert_bands_identical)
from src.losses.dice import DiceLoss, soft_dice_coefficient  # noqa: E402
from src.losses.dcr_loss import (DCRLoss, sobel_levelset_curvature,  # noqa: E402
                                 sobel_magnitude, register_curvature_proxy,
                                 get_curvature_proxy, available_curvature_proxies)
from src.losses.generic_boundary import GenericBoundaryLoss  # noqa: E402
from src.losses.combined import (CombinedLoss, calibrate_beta_by_gradient_norm,  # noqa: E402
                                 gradient_norm_contribution, build_loss_from_config,
                                 assert_shared_band)
from src.models.unet import UNet  # noqa: E402

torch.manual_seed(0)
H = W = 64


def square_mask(size=H, half=18):
    """A square: curvature concentrated at 4 corners, ~0 along straight edges."""
    m = torch.zeros(1, 1, size, size)
    c = size // 2
    m[..., c - half:c + half, c - half:c + half] = 1.0
    return m


def disc_mask(size=H, r=18):
    m = torch.zeros(1, 1, size, size)
    yy, xx = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    c = size // 2
    m[0, 0][((yy - c) ** 2 + (xx - c) ** 2) <= r * r] = 1.0
    return m


# ---- 1. BANDS ARE PIXEL-IDENTICAL --------------------------------------------------
sq = square_mask()
dcr = DCRLoss(band_width=3)
gen = GenericBoundaryLoss(band_width=3)
logits = torch.randn(1, 1, H, W)

band_direct = extract_boundary_band(sq, 3)
w_gen = gen.weight_map(logits, sq)
check("generic weight map IS exactly the shared band (flat weighting)",
      torch.equal(w_gen, band_direct))

dcr0 = DCRLoss(band_width=3, curvature_gamma=0.0, normalise_weights=False)
w_dcr0 = dcr0.weight_map(logits, sq)
maxdiff = (w_dcr0 - band_direct).abs().max().item()
print(f"       max |dCR(gamma=0) band - shared band| = {maxdiff:.3e}")
check(f"dCR at gamma=0 reduces EXACTLY to the shared flat band (maxdiff={maxdiff:.2e})",
      maxdiff < 1e-6)

l_dcr0 = dcr0(logits, sq).item()
l_gen = gen(logits, sq).item()
print(f"       DCRLoss(gamma=0) = {l_dcr0:.8f}   GenericBoundaryLoss = {l_gen:.8f}")
check(f"DCRLoss(gamma=0) == GenericBoundaryLoss numerically "
      f"(|diff|={abs(l_dcr0 - l_gen):.2e}) -- the control is an exact special case",
      abs(l_dcr0 - l_gen) < 1e-6)

# the band both losses use must be identical for the configured widths
try:
    assert_shared_band(dcr, gen, sq)
    shared_ok = True
except AssertionError:
    shared_ok = False
check("assert_shared_band passes for equal widths", shared_ok)
try:
    assert_bands_identical(3, 5, sq)
    caught = False
except AssertionError:
    caught = True
check("assert_bands_identical RAISES on mismatched widths (P5 guard actually fires)", caught)

# ---- 2. dCR WEIGHTS CORNERS > SMOOTH REGIONS ---------------------------------------
print("\n       --- curvature weighting on a synthetic square (corners vs straight edges) ---")
half = 18
c = H // 2
corner_mask = torch.zeros(1, 1, H, W, dtype=torch.bool)
for cy in (c - half, c + half - 1):
    for cx in (c - half, c + half - 1):
        corner_mask[..., cy - 4:cy + 5, cx - 4:cx + 5] = True
edge_mask = torch.zeros(1, 1, H, W, dtype=torch.bool)
edge_mask[..., c - 6:c + 6, c - half - 3:c - half + 4] = True   # left edge midpoint
edge_mask[..., c - half - 3:c - half + 4, c - 6:c + 6] = True   # top edge midpoint

band_sq = extract_boundary_band(sq, 3) > 0

# Raw curvature first: a straight edge has ZERO curvature, a corner does not. This is the
# property the whole weighting rests on, so it is checked directly before the weights.
from src.losses.dcr_loss import gaussian_blur  # noqa: E402
print(f"       {'sigma':>6} {'corner kappa':>13} {'edge kappa':>11}   (level-set curvature)")
for sig in (0.0, 1.0, 2.0):
    k = sobel_levelset_curvature(gaussian_blur(sq, sig))
    ck, ek = k[corner_mask & band_sq].mean().item(), k[edge_mask & band_sq].mean().item()
    print(f"       {sig:>6.1f} {ck:>13.5f} {ek:>11.5f}")
k0 = sobel_levelset_curvature(gaussian_blur(sq, 0.0))
k2 = sobel_levelset_curvature(gaussian_blur(sq, 2.0))
c0, e0 = k0[corner_mask & band_sq].mean().item(), k0[edge_mask & band_sq].mean().item()
c2, e2 = k2[corner_mask & band_sq].mean().item(), k2[edge_mask & band_sq].mean().item()
check(f"UNSMOOTHED curvature does NOT discriminate (corner {c0:.4f} vs edge {e0:.4f}) -- "
      f"differentiating a hard binary mask measures the pixel staircase, not shape", c0 <= e0)
check(f"SMOOTHED (sigma=2) curvature: straight edge is ~zero ({e2:.5f})", e2 < 1e-3)
check(f"SMOOTHED (sigma=2) curvature: corner is clearly non-zero ({c2:.5f})", c2 > 0.05)

results = {}
for proxy in ("sobel_levelset", "sobel_magnitude"):
    d = DCRLoss(band_width=3, curvature_gamma=1.0, curvature_proxy=proxy)
    w = d.weight_map(logits, sq)
    cw = w[corner_mask & band_sq].mean().item()
    ew = w[edge_mask & band_sq].mean().item()
    results[proxy] = (cw, ew, cw / max(ew, 1e-12))
    print(f"       weight  {proxy:<17} corner={cw:.4f}  edge={ew:.4f}  "
          f"ratio={cw / max(ew, 1e-12):.3f}")

cw, ew, ratio = results["sobel_levelset"]
check(f"dCR weights CORNERS more than smooth edges (corner={cw:.4f} > edge={ew:.4f})", cw > ew)
check(f"corner/edge weight ratio is substantial, not marginal (ratio={ratio:.2f} > 1.5)",
      ratio > 1.5)
# gamma bounds the weighting: a maximally curved pixel gets at most (1+gamma)x a straight one.
check(f"weight ratio respects the gamma=1 bound of 2.0 (ratio={ratio:.3f} <= 2.0 + tol)",
      ratio <= 2.0 + 1e-3)
_, _, ratio_mag = results["sobel_magnitude"]
check(f"raw |grad p| does NOT discriminate corners (ratio={ratio_mag:.3f} < "
      f"{ratio:.3f}), which is why it is not the default", ratio_mag < ratio)

# a disc has near-uniform curvature: weights should be far flatter than on a cornered square
d = DCRLoss(band_width=3, curvature_gamma=1.0)
w_disc = d.weight_map(logits, disc_mask())
band_disc = extract_boundary_band(disc_mask(), 3) > 0
cv_disc = (w_disc[band_disc].std() / w_disc[band_disc].mean()).item()
w_sq = d.weight_map(logits, sq)
cv_sq = (w_sq[band_sq].std() / w_sq[band_sq].mean()).item()
print(f"       weight CV: disc(uniform curvature)={cv_disc:.3f}  "
      f"square(corners)={cv_sq:.3f}")
check(f"weights vary MORE on a cornered square than on a smooth disc "
      f"({cv_sq:.3f} > {cv_disc:.3f})", cv_sq > cv_disc)

# ---- 3. CALIBRATION: changes beta AND equalises gradient norms ----------------------
print("\n       --- gradient-norm calibration (B' beta against C's dCR) ---")
model = UNet(encoder_channels=(8, 16), bottleneck_channels=32,
             decoder_channels=(16, 8), init_seed=0)
batches = [{"image": torch.randn(1, 1, 64, 64),
            "mask": square_mask(64, 18)} for _ in range(3)]

g_dcr = gradient_norm_contribution(dcr, model, batches)
g_gen = gradient_norm_contribution(gen, model, batches)
rec = calibrate_beta_by_gradient_norm(dcr, gen, model, batches, reference_beta=1.0)
print(f"       ||grad|| dCR (reference, beta=1.0) = {g_dcr:.6e}")
print(f"       ||grad|| generic (before)          = {g_gen:.6e}")
print(f"       calibrated beta for B'             = {rec['beta']:.6f}")
print(f"       ||grad|| generic * beta (after)    = {g_gen * rec['beta']:.6e}")
print(f"       ratio before = {rec['ratio_before']:.6f}   ratio after = {rec['ratio_after']:.6f}")

check(f"calibration CHANGES beta (is not a no-op): beta={rec['beta']:.6f} != 1.0",
      abs(rec["beta"] - 1.0) > 1e-3)
check(f"calibration EQUALISES gradient norms (ratio after = {rec['ratio_after']:.6f} ~ 1.0)",
      abs(rec["ratio_after"] - 1.0) < 1e-6)
check("ratio before calibration was NOT already 1.0 (so there was a real imbalance)",
      abs(rec["ratio_before"] - 1.0) > 1e-3)
check("calibrated beta is positive and finite",
      rec["beta"] > 0 and torch.isfinite(torch.tensor(rec["beta"])))

# calibration must be deterministic on the same batches
rec2 = calibrate_beta_by_gradient_norm(dcr, gen, model, batches, reference_beta=1.0)
check("calibration is deterministic on identical batches",
      abs(rec["beta"] - rec2["beta"]) < 1e-9)

# and it must leave the model's grads clean
check("calibration leaves no stale gradients on the model",
      all(p.grad is None for p in model.parameters()))

# ---- 4. GRADIENTS FINITE AND NON-NaN ------------------------------------------------
print("\n       --- gradient health ---")
for name, loss_fn in [("DiceLoss", DiceLoss()),
                      ("DCRLoss", DCRLoss(band_width=3)),
                      ("GenericBoundaryLoss", GenericBoundaryLoss(band_width=3)),
                      ("CombinedLoss(dice+dcr)",
                       CombinedLoss(DiceLoss(), DCRLoss(band_width=3), 1.0, 1.0)),
                      ("CombinedLoss(dice+generic)",
                       CombinedLoss(DiceLoss(), GenericBoundaryLoss(band_width=3), 1.0, 1.0))]:
    z = torch.randn(2, 1, H, W, requires_grad=True)
    t = torch.cat([square_mask(), disc_mask()], dim=0)
    out = loss_fn(z, t)
    out.backward()
    finite = bool(torch.isfinite(z.grad).all()) and bool(torch.isfinite(out).all())
    nonzero = bool(z.grad.abs().sum() > 0)
    print(f"       {name:<28} loss={out.item():.6f}  "
          f"grad_norm={z.grad.norm().item():.6e}")
    check(f"{name}: loss and gradient finite, non-NaN", finite)
    check(f"{name}: gradient is non-zero (the loss actually trains something)", nonzero)

# ---- 5. EDGE CASES ------------------------------------------------------------------
print("\n       --- edge cases ---")
empty = torch.zeros(1, 1, H, W)
full = torch.ones(1, 1, H, W)
cases = {
    "empty target (no foreground -> empty band)": empty,
    "full target (no boundary at all)": full,
}
for desc, tgt in cases.items():
    for name, fn in [("dCR", DCRLoss(band_width=3)),
                     ("generic", GenericBoundaryLoss(band_width=3)),
                     ("dice", DiceLoss())]:
        z = torch.randn(1, 1, H, W, requires_grad=True)
        out = fn(z, tgt)
        out.backward()
        ok = bool(torch.isfinite(out)) and bool(torch.isfinite(z.grad).all())
        check(f"{name} finite on {desc}", ok)

# extreme logits must not overflow (BCE-with-logits is the reason this holds)
for scale, desc in [(1e3, "extreme +/-1000 logits"), (0.0, "all-zero logits")]:
    z = (torch.randn(1, 1, H, W) * scale).requires_grad_(True)
    for name, fn in [("dCR", DCRLoss(band_width=3)),
                     ("generic", GenericBoundaryLoss(band_width=3)),
                     ("dice", DiceLoss())]:
        out = fn(z, sq)
        out.backward()
        check(f"{name} finite with {desc}",
              bool(torch.isfinite(out)) and bool(torch.isfinite(z.grad).all()))
        z.grad = None

# perfect prediction: dice loss ~0
perfect = (sq * 40.0 - 20.0)
check(f"DiceLoss ~0 on a perfect prediction ({DiceLoss()(perfect, sq).item():.2e})",
      DiceLoss()(perfect, sq).item() < 1e-3)
check("empty-vs-empty Dice coefficient is 1.0 (not 0/0)",
      abs(soft_dice_coefficient(torch.zeros(1, 1, 8, 8),
                                torch.zeros(1, 1, 8, 8)).item() - 1.0) < 1e-6)

# ---- 6. SWAPPABILITY: the real equation is a one-function change ---------------------
print("\n       --- surrogate swappability (A06) ---")
check("registry exposes both proxies",
      set(available_curvature_proxies()) >= {"sobel_levelset", "sobel_magnitude"})


def _fake_official(mask, eps=1e-6):
    return torch.ones_like(mask) * 0.5


register_curvature_proxy("official_test", _fake_official)
d_off = DCRLoss(band_width=3, curvature_proxy="official_test")
check("a newly registered proxy is usable with no other code change",
      get_curvature_proxy("official_test") is _fake_official
      and torch.isfinite(d_off(logits, sq)))
try:
    register_curvature_proxy("official_test", _fake_official)
    guarded = False
except ValueError:
    guarded = True
check("registry refuses silent overwrite of an existing proxy", guarded)

# ---- 7. CONFIG-DRIVEN CONSTRUCTION ---------------------------------------------------
import yaml  # noqa: E402

print("\n       --- config-driven construction ---")
cfg_c = yaml.safe_load((ROOT / "configs" / "model_c_pruned_dcr.yaml").read_text(encoding="utf-8"))
cfg_bp = yaml.safe_load((ROOT / "configs" / "model_b_prime_generic.yaml").read_text(encoding="utf-8"))
cfg_b = yaml.safe_load((ROOT / "configs" / "model_b_pruned.yaml").read_text(encoding="utf-8"))

loss_c = build_loss_from_config(cfg_c)
check("Model C builds a CombinedLoss with a dCR boundary term",
      isinstance(loss_c.boundary_loss, DCRLoss))
loss_b = build_loss_from_config(cfg_b)
check("Model B builds with NO boundary term", loss_b.boundary_loss is None)
try:
    build_loss_from_config(cfg_bp)
    raised = False
except ValueError:
    raised = True
check("Model B' REFUSES to build while beta is null -- it must be calibrated first, "
      "not silently trained with beta=0", raised)
check("C's configured band_width equals B''s (master §8)",
      cfg_c["loss"]["boundary"]["band_width"] == cfg_bp["loss"]["boundary"]["band_width"])

# ---- 8. source labelling -------------------------------------------------------------
src = (ROOT / "src" / "losses" / "dcr_loss.py").read_text(encoding="utf-8")
check("dcr_loss.py carries the [ASSUMPTION] A06 surrogate warning",
      "SURROGATE" in src and "A06" in src)
check("dcr_loss.py distinguishes itself from bending loss (Wang 2020)",
      "Wang 2020" in src and "bending loss" in src.lower())
check("dcr_loss.py distinguishes itself from the OCT curvature loss (Xing 2022)",
      "Xing 2022" in src)
check("dcr_loss.py documents the one-function replacement path",
      "register_curvature_proxy" in src and "one function" in src.lower())
check("no numpy in the losses' forward path (master §8)",
      not any("import numpy" in (ROOT / "src" / "losses" / f).read_text(encoding="utf-8")
              for f in ("dcr_loss.py", "generic_boundary.py", "dice.py",
                        "boundary_band.py", "combined.py")))

r = subprocess.run([sys.executable, "-m", "compileall", "-q", str(ROOT / "src" / "losses")],
                   capture_output=True, text=True)
check(f"src/losses byte-compiles (exit {r.returncode})", r.returncode == 0)

print()
print("=" * 78)
if failures:
    print(f"RESULT: FAIL ({len(failures)} check(s) failed)")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("RESULT: PASS — Phase 5 verification complete.")
print("=" * 78)
print(f"Bands pixel-identical; dCR corner/edge weight ratio {ratio:.2f}; calibration moved")
print(f"beta 1.0 -> {rec['beta']:.4f} and equalised gradient norms to {rec['ratio_after']:.6f}.")
