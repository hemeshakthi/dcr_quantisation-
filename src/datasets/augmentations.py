"""Phase 2 step 2: seeded, deterministic augmentation.

Implemented directly on cv2/numpy rather than via albumentations. Master §10 requires FULL
determinism and a fixed augmentation seed SHARED across B/B'/C (only model_init_seed may vary),
and the augmented stream must be byte-identical between this Windows host and Kaggle's Linux
runners. The pinned albumentations (1.3.1, master §16) is not what is installed here (2.0.8),
and the 2.x rewrite changed both transform signatures and internal RNG handling -- so relying on
it would silently produce a DIFFERENT augmented stream on the two machines, which would break
the B/B'/C comparison at its foundation. A small explicit implementation removes that risk
entirely. [ASSUMPTION A23]

Every transform is driven by an explicit `numpy.random.Generator`, so the same
(seed, epoch, index) always yields the same output. Nothing reads global RNG state.

Masks are ALWAYS resampled with INTER_NEAREST so instance IDs are never blended (A21).

Usage:
    aug = Augmenter(seed=1234)
    img_a, bin_a, inst_a = aug(img, binary, instances, epoch=0, index=17)
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import cv2
import numpy as np

# ----------------------------------------------------------------------------------
# [ASSUMPTION A23] Augmentation policy.
# The paper's augmentation policy is unrecoverable (paywalled body, A03). This is the
# conservative, standard policy for microscopy cell segmentation:
#   * flips + 90-degree rotations  -- exactly label-preserving; cells have no canonical
#     orientation, so these are "free" symmetries of the data
#   * small rotation/scale/shift   -- mild affine; cells are near-isotropic
#   * MILD elastic deformation     -- the U-Net paper's own augmentation for microscopy
#     (Ronneberger 2015). Deliberately mild: this study MEASURES boundary curvature, and a
#     strong elastic field would itself manufacture curvature and contaminate the very
#     signal under test (master §11 bins images by curvature).
#   * light intensity jitter/noise -- brightness/contrast/gaussian noise, matched to the
#     low-contrast boundaries the paper targets
# NO vertical-only asymmetric ops, NO cutout/erasing (would fabricate false boundaries),
# NO heavy elastic. Probabilities and magnitudes are all explicit below.
# ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class AugmentationConfig:
    p_hflip: float = 0.5
    p_vflip: float = 0.5
    p_rot90: float = 0.5

    p_affine: float = 0.5
    max_rotation_deg: float = 15.0
    max_scale_delta: float = 0.10      # +/-10%
    max_shift_frac: float = 0.05       # +/-5% of side

    p_elastic: float = 0.25            # mild + infrequent, see note above
    elastic_alpha: float = 12.0        # displacement magnitude in px
    elastic_sigma: float = 16.0        # smoothing of the displacement field (large = smooth)

    p_brightness_contrast: float = 0.5
    max_brightness_delta: float = 0.10
    max_contrast_delta: float = 0.10

    p_gaussian_noise: float = 0.25
    noise_std: float = 0.02

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_CONFIG = AugmentationConfig()


class Augmenter:
    """Deterministic augmenter. Output depends only on (seed, epoch, index) and the config."""

    def __init__(self, seed: int, config: AugmentationConfig = DEFAULT_CONFIG):
        self.seed = int(seed)
        self.config = config

    def _rng(self, epoch: int, index: int) -> np.random.Generator:
        # SeedSequence spawn-free derivation: a stable, collision-resistant stream per
        # (seed, epoch, index). Reproducible across platforms and numpy versions.
        return np.random.default_rng([self.seed, int(epoch), int(index)])

    # ---------------- individual transforms ----------------
    @staticmethod
    def _apply_affine(img, binary, inst, M, size):
        img = cv2.warpAffine(img, M, size, flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT_101)
        binary = cv2.warpAffine(binary, M, size, flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_REFLECT_101)
        inst = cv2.warpAffine(inst, M, size, flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_REFLECT_101)
        return img, binary, inst

    @staticmethod
    def _elastic(img, binary, inst, rng, alpha, sigma):
        h, w = img.shape[:2]
        # smooth random displacement field
        dx = rng.random((h, w), dtype=np.float32) * 2.0 - 1.0
        dy = rng.random((h, w), dtype=np.float32) * 2.0 - 1.0
        ks = int(max(3, (int(sigma) * 4 + 1) | 1))
        cv2.GaussianBlur(dx, (ks, ks), sigma, dst=dx)
        cv2.GaussianBlur(dy, (ks, ks), sigma, dst=dy)
        # normalise so alpha is the true peak displacement in px
        for d in (dx, dy):
            m = float(np.abs(d).max())
            if m > 0:
                d *= (alpha / m)
        gx, gy = np.meshgrid(np.arange(w, dtype=np.float32),
                             np.arange(h, dtype=np.float32))
        mx = gx + dx
        my = gy + dy
        del dx, dy, gx, gy
        img = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        binary = cv2.remap(binary, mx, my, cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_REFLECT_101)
        inst = cv2.remap(inst, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101)
        del mx, my
        return img, binary, inst

    # ---------------- entry point ----------------
    def __call__(self, img: np.ndarray, binary: np.ndarray, inst: np.ndarray,
                 epoch: int = 0, index: int = 0):
        """Augment one (image, binary mask, instance mask) triple.

        Geometric ops are applied identically to all three arrays; photometric ops touch the
        image only. Returns new arrays; inputs are not modified.
        """
        c = self.config
        rng = self._rng(epoch, index)

        img = np.array(img, dtype=np.float32, copy=True)
        binary = np.array(binary, dtype=np.uint8, copy=True)
        # int32 keeps room for instance IDs during warping; cast back at the end
        inst = np.array(inst, dtype=np.int32, copy=True)
        h, w = img.shape[:2]

        if rng.random() < c.p_hflip:
            img, binary, inst = img[:, ::-1], binary[:, ::-1], inst[:, ::-1]
        if rng.random() < c.p_vflip:
            img, binary, inst = img[::-1], binary[::-1], inst[::-1]
        if rng.random() < c.p_rot90:
            k = int(rng.integers(1, 4))
            img = np.rot90(img, k)
            binary = np.rot90(binary, k)
            inst = np.rot90(inst, k)

        img = np.ascontiguousarray(img)
        binary = np.ascontiguousarray(binary)
        inst = np.ascontiguousarray(inst)

        if rng.random() < c.p_affine:
            angle = float(rng.uniform(-c.max_rotation_deg, c.max_rotation_deg))
            scale = 1.0 + float(rng.uniform(-c.max_scale_delta, c.max_scale_delta))
            tx = float(rng.uniform(-c.max_shift_frac, c.max_shift_frac)) * w
            ty = float(rng.uniform(-c.max_shift_frac, c.max_shift_frac)) * h
            M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
            M[0, 2] += tx
            M[1, 2] += ty
            img, binary, inst = self._apply_affine(img, binary, inst, M, (w, h))

        if rng.random() < c.p_elastic:
            img, binary, inst = self._elastic(img, binary, inst, rng,
                                              c.elastic_alpha, c.elastic_sigma)

        if rng.random() < c.p_brightness_contrast:
            b = float(rng.uniform(-c.max_brightness_delta, c.max_brightness_delta))
            g = 1.0 + float(rng.uniform(-c.max_contrast_delta, c.max_contrast_delta))
            mean = float(img.mean())
            img = (img - mean) * g + mean + b

        if rng.random() < c.p_gaussian_noise:
            img = img + rng.normal(0.0, c.noise_std, img.shape).astype(np.float32)

        binary = (inst > 0).astype(np.uint8)  # keep binary exactly the instance support
        return (np.ascontiguousarray(img, dtype=np.float32),
                np.ascontiguousarray(binary, dtype=np.uint8),
                np.ascontiguousarray(inst.astype(np.uint16)))


def self_test(verbose: bool = True) -> bool:
    """Determinism + invariant checks. Returns True if all pass."""
    import hashlib

    rng = np.random.default_rng(0)
    img = rng.random((128, 128), dtype=np.float32)
    inst = np.zeros((128, 128), np.uint16)
    for i, (cy, cx) in enumerate([(30, 30), (30, 90), (90, 30), (90, 95)], start=1):
        yy, xx = np.ogrid[:128, :128]
        inst[(yy - cy) ** 2 + (xx - cx) ** 2 <= 12 ** 2] = i
    binary = (inst > 0).astype(np.uint8)

    def h(a):
        return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()[:12]

    ok = True
    aug = Augmenter(seed=1234)

    # 1. same (seed, epoch, index) -> identical output, across repeated calls
    a1 = aug(img, binary, inst, epoch=0, index=5)
    a2 = aug(img, binary, inst, epoch=0, index=5)
    same = all(h(x) == h(y) for x, y in zip(a1, a2))
    ok &= same
    if verbose:
        print(f"  [{'PASS' if same else 'FAIL'}] repeatable for identical (seed, epoch, index)")

    # 2. a fresh Augmenter with the same seed reproduces it (no hidden state)
    a3 = Augmenter(seed=1234)(img, binary, inst, epoch=0, index=5)
    same2 = all(h(x) == h(y) for x, y in zip(a1, a3))
    ok &= same2
    if verbose:
        print(f"  [{'PASS' if same2 else 'FAIL'}] fresh instance, same seed -> same output")

    # 3. different index -> different output (augmentation actually varies)
    a4 = aug(img, binary, inst, epoch=0, index=6)
    diff = h(a1[0]) != h(a4[0])
    ok &= diff
    if verbose:
        print(f"  [{'PASS' if diff else 'FAIL'}] different index -> different output")

    # 4. different seed -> different output
    a5 = Augmenter(seed=999)(img, binary, inst, epoch=0, index=5)
    diff2 = h(a1[0]) != h(a5[0])
    ok &= diff2
    if verbose:
        print(f"  [{'PASS' if diff2 else 'FAIL'}] different seed -> different output")

    # 5. binary must remain exactly the instance support
    inv = np.array_equal(a1[1] > 0, a1[2] > 0)
    ok &= inv
    if verbose:
        print(f"  [{'PASS' if inv else 'FAIL'}] binary mask == instance support after augment")

    # 6. instance IDs are never blended into new values
    before = set(np.unique(inst).tolist())
    after = set(np.unique(a1[2]).tolist())
    nonew = after.issubset(before)
    ok &= nonew
    if verbose:
        print(f"  [{'PASS' if nonew else 'FAIL'}] no invented instance IDs "
              f"(before={sorted(before)}, after={sorted(after)})")

    # 7. shapes preserved
    shp = a1[0].shape == img.shape and a1[1].shape == img.shape and a1[2].shape == img.shape
    ok &= shp
    if verbose:
        print(f"  [{'PASS' if shp else 'FAIL'}] shapes preserved")

    # 8. dtypes as documented
    dt = (a1[0].dtype == np.float32 and a1[1].dtype == np.uint8 and a1[2].dtype == np.uint16)
    ok &= dt
    if verbose:
        print(f"  [{'PASS' if dt else 'FAIL'}] dtypes float32/uint8/uint16")

    return bool(ok)


if __name__ == "__main__":
    print("augmentation self-test (determinism + invariants)")
    print(f"  config: {DEFAULT_CONFIG.to_dict()}")
    raise SystemExit(0 if self_test() else 1)
