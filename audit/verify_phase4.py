"""Phase 4 verification: the frozen plain U-Net (master §7).

Checks the required contract -- forward (2,1,256,256)->(2,1,256,256) with negative values
present (proving no output activation), parameter count, skip-connection shape trace, and a
scope grep -- plus the architectural invariants master §7 freezes.
"""
import gc
import re
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
print("PHASE 4 VERIFICATION -- plain U-Net (master §7, FROZEN)")
print("=" * 78)

from src.models.unet import UNet, DoubleConv, Up, DOWNSAMPLE_FACTOR  # noqa: E402

# ---- 1. master §7 literal spelling ------------------------------------------------
# encoder [64,128,256,512,1024] + decoder [512,256,128,64] is only consistent if the
# trailing 1024 is the bottleneck. Both spellings must yield the same network.
m_literal = UNet(encoder_channels=(64, 128, 256, 512, 1024),
                 bottleneck_channels=1024,
                 decoder_channels=(512, 256, 128, 64))
m = UNet(encoder_channels=(64, 128, 256, 512),
         bottleneck_channels=1024,
         decoder_channels=(512, 256, 128, 64))
check("master §7's literal encoder [64,128,256,512,1024] is accepted",
      m_literal.encoder_channels == (64, 128, 256, 512)
      and m_literal.bottleneck_channels == 1024)
check("both spellings build an identical network (same parameter count)",
      m_literal.count_parameters() == m.count_parameters())
# Released immediately: each full model is ~124 MB of parameters and this host's commit limit
# equals physical RAM (A18). Holding all of them alive was intermittently segfaulting the audit.
del m_literal
gc.collect()
check("encoder widths are (64,128,256,512)", m.encoder_channels == (64, 128, 256, 512))
check("bottleneck is 1024", m.bottleneck_channels == 1024)
check("decoder widths are (512,256,128,64)", m.decoder_channels == (512, 256, 128, 64))
check("4 pooling stages -> input divisibility 16",
      len(m.encoders) == 4 and DOWNSAMPLE_FACTOR == 16)

# ---- 2. head: Conv2d(64,1,1) with NO activation -----------------------------------
head = m.head
check("head is Conv2d with kernel_size 1x1",
      isinstance(head, torch.nn.Conv2d) and head.kernel_size == (1, 1))
check("head maps 64 -> 1 channels",
      head.in_channels == 64 and head.out_channels == 1)
last = list(m.children())[-1]
check("head is the FINAL module -- nothing follows it (no activation)",
      isinstance(last, torch.nn.Conv2d))
acts = [type(mod).__name__ for mod in m.modules()
        if isinstance(mod, (torch.nn.Sigmoid, torch.nn.Softmax, torch.nn.Tanh))]
check(f"no Sigmoid/Softmax/Tanh anywhere in the module tree (found {acts})", not acts)

# ---- 3. forward: (2,1,256,256) -> (2,1,256,256), and it must go NEGATIVE -----------
m.eval()
x = torch.randn(2, 1, 256, 256)
with torch.no_grad():
    y = m(x)
print(f"       input  {tuple(x.shape)}  ->  output {tuple(y.shape)}")
print(f"       output range: min={y.min().item():.4f}  max={y.max().item():.4f}  "
      f"mean={y.mean().item():.4f}")
check("forward (2,1,256,256) -> (2,1,256,256)", tuple(y.shape) == (2, 1, 256, 256))
check("output dtype is float32", y.dtype == torch.float32)
# The decisive activation test: a sigmoid/ReLU head could never emit a negative value.
check(f"output contains NEGATIVE values -> raw logits, no activation "
      f"(min={y.min().item():.4f})", bool((y < 0).any()))
check("output is not bounded to [0,1] (would indicate a sigmoid)",
      bool(y.min() < 0.0 or y.max() > 1.0))
check("output contains no NaN/Inf", bool(torch.isfinite(y).all()))

# gradients must flow back to the very first conv
m.train()
xg = torch.randn(2, 1, 64, 64, requires_grad=True)
m(xg).sum().backward()
first_conv = m.encoders[0].block[0]
check("gradient reaches the first encoder conv (network is fully connected end to end)",
      first_conv.weight.grad is not None and bool(first_conv.weight.grad.abs().sum() > 0))

# ---- 4. parameter count -----------------------------------------------------------
n_params = m.count_parameters(trainable_only=False)
n_train = m.count_parameters(trainable_only=True)
print(f"\n       PARAMETER COUNT: {n_params:,} total, {n_train:,} trainable")
print(f"       ({n_params / 1e6:.2f} M)")
check(f"parameter count is in the expected ~31M range for a standard U-Net "
      f"(got {n_params:,})", 25_000_000 < n_params < 40_000_000)
check("all parameters are trainable (no frozen/pretrained weights)", n_params == n_train)

# ---- 5. skip-connection shape trace ------------------------------------------------
print("\n       SKIP-CONNECTION SHAPE TRACE  (input 2x1x256x256)")
print("       " + "-" * 66)
trace = m.trace_shapes(torch.randn(2, 1, 256, 256))
for t in trace:
    extra = ""
    if t.get("skip"):
        extra = "   <-- kept as skip"
    if t.get("skip_from"):
        extra = f"   <-- concat skip {t['skip_from']}"
    print(f"       {t['stage']:<12} {str(t['shape']):<22}{extra}")

skips = [t for t in trace if t.get("skip")]
decs = [t for t in trace if t.get("skip_from")]
check(f"4 encoder skips captured (got {len(skips)})", len(skips) == 4)
check(f"4 decoder stages consume a skip (got {len(decs)})", len(decs) == 4)
# Each decoder must consume the encoder skip at the MATCHING resolution, deepest first.
ok_pairing = True
for dec, skip in zip(decs, reversed(skips)):
    if dec["skip_from"] != skip["shape"] or dec["shape"][-2:] != skip["shape"][-2:]:
        ok_pairing = False
check("each decoder stage concatenates the encoder skip at the matching resolution "
      "(deepest first)", ok_pairing)
check("encoder resolutions halve: 256 -> 128 -> 64 -> 32",
      [t["shape"][-1] for t in skips] == [256, 128, 64, 32])
check("encoder widths deepen: 64 -> 128 -> 256 -> 512",
      [t["shape"][1] for t in skips] == [64, 128, 256, 512])
check("bottleneck is 1024 channels at 16x16",
      [t for t in trace if t["stage"] == "bottleneck"][0]["shape"][1:] == (1024, 16, 16))
check("decoder restores full resolution 256x256",
      trace[-1]["shape"][-2:] == (256, 256))

# ---- 6. one class serves all five variants -----------------------------------------
import yaml  # noqa: E402

# Only the parameter COUNT is retained per variant, never the model itself (A18).
param_counts = {}
for name in ["model_vanilla.yaml", "model_a_dcr.yaml", "model_b_pruned.yaml",
             "model_b_prime_generic.yaml", "model_c_pruned_dcr.yaml"]:
    cfg = yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))
    try:
        built = UNet.from_config(cfg)
        param_counts[name] = built.count_parameters()
        del built
        gc.collect()
        ok = True
    except Exception as e:
        print(f"       {name}: {type(e).__name__}: {e}")
        ok = False
    check(f"UNet.from_config builds {name}", ok)
check("all five variants use the SAME class with identical architecture "
      "(architecture is never a variable, master §7)",
      len(set(param_counts.values())) == 1)

# from_config must refuse anything that violates the frozen spec
for bad, why in [({"model": {"architecture": "attention_unet"}}, "non-plain architecture"),
                 ({"model": {"final_activation": "sigmoid"}}, "an output activation")]:
    try:
        UNet.from_config(bad)
        rejected = False
    except ValueError:
        rejected = True
    check(f"from_config REJECTS {why}", rejected)

# input-size contract
try:
    m(torch.randn(1, 1, 100, 100))
    raised = False
except ValueError:
    raised = True
check("non-divisible-by-16 input raises a clear error rather than failing deep in the decoder",
      raised)

# ---- 7. seeding: model_init_seed is the ONLY thing that varies across seeds ---------
# Tiny models here: seeding behaviour is width-independent, and building three more full
# 31M-param networks is what tipped this audit into a segfault under A18 memory pressure.
_kw = dict(encoder_channels=(4, 8), bottleneck_channels=16, decoder_channels=(8, 4))
a = UNet(init_seed=1, **_kw)
b = UNet(init_seed=1, **_kw)
c = UNet(init_seed=2, **_kw)
same = all(torch.equal(p, q) for p, q in zip(a.state_dict().values(),
                                             b.state_dict().values()))
diff = any(not torch.equal(p, q) for p, q in zip(a.state_dict().values(),
                                                 c.state_dict().values()))
del a, b, c
gc.collect()
check("same init_seed -> byte-identical weights (reproducible, master §10)", same)
check("different init_seed -> different weights (seeds are real, not cosmetic)", diff)
pre = torch.randn(3)
torch.manual_seed(999)
before = torch.randn(3)
torch.manual_seed(999)
UNet(init_seed=7, **_kw)
after = torch.randn(3)
check("constructing a seeded model does NOT disturb the caller's RNG stream "
      "(so the shared augmentation seed is unaffected)", torch.equal(before, after))

# ---- 8. prunable layer names -------------------------------------------------------
names = m.prunable_conv_names()
print(f"\n       {len(names)} prunable conv layers (head excluded)")
check("prunable list is non-empty", len(names) > 0)
check("the 1x1 output head is NOT prunable (would change the task, not the capacity)",
      not any(n == "head" for n in names))
check("every prunable name resolves to a real Conv2d module",
      all(isinstance(dict(m.named_modules())[n], torch.nn.Conv2d) for n in names))

# ---- 9. scope grep -----------------------------------------------------------------
src = (ROOT / "src" / "models" / "unet.py").read_text(encoding="utf-8")
FORBIDDEN = ["attention", "transformer", "pretrained", "unet++", "unetpp",
             "distill", "student", "teacher"]
violations = []
for term in FORBIDDEN:
    for ln, line in enumerate(src.splitlines(), 1):
        if term not in line.lower():
            continue
        # Prose EXCLUDING these is mandated by master §7; only real code would be a violation.
        if re.search(rf"^\s*(import|from)\s+.*{term}", line, re.I) or \
           re.search(rf"^\s*(class|def)\s+\w*{term}", line, re.I) or \
           re.search(rf"nn\.\w*{term}", line, re.I):
            violations.append(f"line {ln}: {line.strip()[:80]}")
check(f"no attention/transformer/pretrained/UNet++/distillation CODE in unet.py "
      f"({violations or 'none'})", not violations)
check("the 'why plain U-Net' rationale is present as a docstring (master §7)",
      "WHY PLAIN U-NET" in src and "Attention U-Net is excluded" in src
      and "UNet++ is excluded" in src)
check("channel widths carry the [ASSUMPTION] A08 label",
      "[ASSUMPTION] A08" in src and "confirm" in src.lower())
check("no torchvision/timm/pretrained-weight imports",
      not re.search(r"^\s*(import|from)\s+(torchvision|timm)", src, re.M))

r = subprocess.run([sys.executable, "-m", "compileall", "-q",
                    str(ROOT / "src" / "models")], capture_output=True, text=True)
check(f"src/models byte-compiles (exit {r.returncode})", r.returncode == 0)

print()
print("=" * 78)
if failures:
    print(f"RESULT: FAIL ({len(failures)} check(s) failed)")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("RESULT: PASS — Phase 4 verification complete.")
print("=" * 78)
print(f"Single frozen backbone, {n_params:,} parameters, raw-logit output.")
print("UNet.from_channel_mask is implemented (backend-authoring phase) alongside")
print("src.compression.structured_pruning, which defines the mask semantics it consumes --")
print("see audit/verify_phase7.py and audit/verify_backend_integration.py.")
