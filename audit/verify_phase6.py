"""Phase 6 verification: training pipeline, via a TINY SYNTHETIC CPU smoke test.

DELIBERATELY SYNTHETIC. This host is CPU-only with its Windows commit limit equal to physical
RAM (A18) -- a 4 MiB allocation has been observed to fail. The smoke test therefore uses
2x1x64x64 tensors held in memory and NEVER touches the real 1024x1024 / 1400x1400 arrays; those
belong to Kaggle (master §17). What is being verified here is the TRAINING MACHINERY -- does
loss decrease, does resume reproduce, is a seed reproducible -- not model quality, which cannot
be judged from a toy tensor and is not claimed to be.

Required checks:
  * loss decreases
  * checkpoint + CSV + experiment_log row written, with a git hash
  * resume identical
  * two same-seed runs identical
Plus the sigma-drift guard: curvature_smooth_sigma must come from the YAML.
"""
from __future__ import annotations

import csv
import shutil
import subprocess
import sys
import tempfile
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
print("PHASE 6 VERIFICATION -- training pipeline (tiny synthetic CPU smoke test)")
print("=" * 78)

from src.config import (compose_run_config, config_fingerprint,  # noqa: E402
                        validate_config, load_yaml)
from src.losses.combined import build_loss_from_config  # noqa: E402
from src.models.unet import UNet  # noqa: E402
from src.training.checkpointing import (BestCheckpointTracker, checkpoint_hash,  # noqa: E402
                                        load_checkpoint)
from src.training.experiment_log import (append_run, build_run_record,  # noqa: E402
                                         git_commit_hash, audit_completeness)
from src.training.seed_manager import (SeedSet, apply_run_determinism,  # noqa: E402
                                       seeds_for_run, set_determinism,
                                       rng_state_snapshot, rng_state_restore)
from src.training.trainer import Trainer  # noqa: E402
from src.training.validator import validate, monitored_value  # noqa: E402
from src.compression.bn_recalibration import (recalibrate_batchnorm,  # noqa: E402
                                              reset_bn_statistics,
                                              batch_order_fingerprint,
                                              assert_identical_recalibration)

# ----------------------------------------------------------------------------------
# Tiny synthetic corpus: 2 images, 64x64. Never real data (A18, master §17).
# ----------------------------------------------------------------------------------
IMG, N = 64, 4


def synth_batches(n_images=N, size=IMG, seed=0):
    g = torch.Generator().manual_seed(seed)
    items = []
    for i in range(n_images):
        mask = torch.zeros(1, size, size)
        # a square: learnable, and it has corners so the dCR weighting is exercised
        a, b = 16 + (i % 3) * 2, 48 - (i % 3) * 2
        mask[:, a:b, a:b] = 1.0
        img = mask * 1.5 + torch.randn(1, size, size, generator=g) * 0.3
        items.append({"image": img, "mask": mask, "image_id": f"synth_{i:02d}",
                      "meta": {"dataset": "synthetic", "split": "train"}})
    return items


class TinyLoader:
    """Minimal deterministic loader: fixed order, batch size 2. No workers, no shuffling."""

    def __init__(self, items, batch_size=2):
        self.items = items
        self.batch_size = batch_size
        self.dataset = self

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return (len(self.items) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        for i in range(0, len(self.items), self.batch_size):
            chunk = self.items[i:i + self.batch_size]
            yield {"image": torch.stack([c["image"] for c in chunk]),
                   "mask": torch.stack([c["mask"] for c in chunk]),
                   "image_id": [c["image_id"] for c in chunk],
                   "meta": [c["meta"] for c in chunk]}


def tiny_model(seed=1):
    return UNet(encoder_channels=(4, 8), bottleneck_channels=16,
                decoder_channels=(8, 4), init_seed=seed)


def tiny_cfg(max_epochs=6, tmp=None):
    cfg = compose_run_config("model_a_dcr.yaml")
    cfg["training"].update(max_epochs=max_epochs, batch_size=2,
                           gradient_accumulation_steps=1, mixed_precision=False)
    cfg["optimizer"]["lr"] = 5e-3          # smoke test only: make movement visible in 6 epochs
    cfg["early_stopping"].update(enabled=False, patience=999)
    cfg["scheduler"]["patience"] = 99
    cfg["logging"]["tensorboard"] = False
    cfg["model"].update(encoder_channels=[4, 8], bottleneck_channels=16,
                        decoder_channels=[8, 4])
    return cfg


# ---- 1. determinism + seed separation ---------------------------------------------
cfg = tiny_cfg()
s0 = seeds_for_run(cfg, 0)
s1 = seeds_for_run(cfg, 1)
s2 = seeds_for_run(cfg, 2)
print(f"       seeds: {[s.as_dict() for s in (s0, s1, s2)]}")
check("3 distinct model_init_seeds (master §10)",
      len({s.model_init_seed for s in (s0, s1, s2)}) == 3)
check("augmentation seed is SHARED across all 3 seeds -- B/B'/C must see identical data",
      len({s.augmentation_seed for s in (s0, s1, s2)}) == 1)
rec = set_determinism(123)
check("set_determinism enables cudnn.deterministic and disables benchmark",
      torch.backends.cudnn.deterministic is True and torch.backends.cudnn.benchmark is False)
check("determinism record reports whether deterministic algorithms were actually enabled",
      "use_deterministic_algorithms" in rec)
snap = rng_state_snapshot()
a = torch.randn(3)
rng_state_restore(snap)
b = torch.randn(3)
check("RNG snapshot/restore reproduces the stream exactly (resume depends on this)",
      torch.equal(a, b))

# ---- 2. config: sigma comes from YAML and sigma<=0 is rejected ---------------------
print("\n       --- config-driven sigma (the Phase 5 drift guard) ---")
for name in ("model_a_dcr.yaml", "model_c_pruned_dcr.yaml"):
    raw = load_yaml(name)
    sig = raw["loss"]["boundary"]["curvature_smooth_sigma"]
    print(f"       {name}: curvature_smooth_sigma = {sig}")
    check(f"{name} declares curvature_smooth_sigma = 2.0 in the YAML", float(sig) == 2.0)
composed = compose_run_config("model_c_pruned_dcr.yaml")
check("composed config carries sigma=2.0 through to training",
      float(composed["loss"]["boundary"]["curvature_smooth_sigma"]) == 2.0)
bad = compose_run_config("model_a_dcr.yaml")
bad["loss"]["boundary"]["curvature_smooth_sigma"] = 0.0
check("validate_config REJECTS a dCR run with sigma=0 (would reintroduce the Phase 5 bug)",
      any("curvature_smooth_sigma" in p for p in validate_config(bad)))
missing = compose_run_config("model_a_dcr.yaml")
del missing["loss"]["boundary"]["curvature_smooth_sigma"]
check("validate_config REJECTS a dCR run with sigma missing entirely",
      any("curvature_smooth_sigma" in p for p in validate_config(missing)))
check("composed config passes validation as shipped",
      not validate_config(compose_run_config("model_a_dcr.yaml")))

# ---- 3. the smoke run: loss decreases, artefacts written --------------------------
print("\n       --- tiny synthetic training run (2x1x64x64) ---")
tmp = Path(tempfile.mkdtemp(prefix="dcr_phase6_"))
items = synth_batches()
loader = TinyLoader(items)


def run(tag, seed=1, max_epochs=6, resume_from=None, out=None):
    set_determinism(seed)
    cfg = tiny_cfg(max_epochs)
    fp = config_fingerprint(cfg)
    model = tiny_model(seed)
    loss_fn = build_loss_from_config(cfg)
    t = Trainer(model=model, loss_fn=loss_fn, train_loader=loader, val_loader=loader,
                config=cfg, run_id=tag, device="cpu", output_root=out or tmp,
                config_fingerprint=fp, resume_from=resume_from, log_tensorboard=False)
    return t.fit(), cfg, fp


summary, cfg_used, fp_used = run("smoke_a", seed=1, max_epochs=6)
csv_path = Path(summary["csv_log"])
rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
losses = [float(r["train_loss"]) for r in rows]
print(f"       train_loss by epoch: {[round(x, 4) for x in losses]}")
check(f"CSV log written with one row per epoch ({len(rows)} rows)", len(rows) == 6)
check(f"LOSS DECREASES: first {losses[0]:.4f} -> last {losses[-1]:.4f}", losses[-1] < losses[0])
check("gradient norm logged every epoch (master §10)",
      all(r.get("grad_norm") not in (None, "") for r in rows))
check("learning rate logged every epoch", all("lr" in r for r in rows))
check("monitored metric is val boundary F1, not Dice (P1 rationale)",
      rows[0]["monitor"].startswith("val_boundary_f1"))
best, final = Path(summary["checkpoint_best_path"]), Path(summary["checkpoint_final_path"])
check("BEST checkpoint written", best.exists())
check("FINAL checkpoint written", final.exists())
ck = load_checkpoint(best, map_location="cpu")
check("checkpoint stores config, fingerprint and RNG state (resume needs all three)",
      bool(ck.get("config")) and ck.get("config_fingerprint") == fp_used
      and ck.get("rng_state") is not None)
check("mean gradient norm reported in the run summary",
      summary.get("mean_gradient_norm") not in (None, ""))

# experiment_log row + git hash
log_path = tmp / "results" / "tables" / "experiment_log.csv"
shutil.copy(ROOT / "results" / "tables" / "experiment_log.csv",
            log_path.parent.mkdir(parents=True, exist_ok=True) or log_path)
record = build_run_record("smoke_a", cfg_used, summary, config_fingerprint=fp_used,
                          seeds=SeedSet(1, 1234, 1234).as_dict(), executed_on="local-smoke")
append_run(record, log_path)
logged = list(csv.DictReader(log_path.open(newline="", encoding="utf-8")))
check("experiment_log row appended", len(logged) == 1)
row = logged[0]
print(f"       git_commit_hash = {row['git_commit_hash']}")
check("experiment_log row records a git commit hash", bool(row["git_commit_hash"])
      and row["git_commit_hash"] != "UNKNOWN")
check("experiment_log row records the seed separately (master §10)",
      row["model_init_seed"] == "1")
check("experiment_log row records mixed_precision = False (master §10)",
      str(row["mixed_precision"]) in ("False", "false"))
check(f"experiment_log row records curvature_smooth_sigma "
      f"(={row['curvature_smooth_sigma']}) so a drift is detectable",
      float(row["curvature_smooth_sigma"]) == 2.0)
check("experiment_log row records the config fingerprint",
      row["config_fingerprint"] == fp_used)
try:
    append_run({**record, "not_a_column": 1}, log_path)
    refused = False
except ValueError:
    refused = True
check("append_run raises on an unknown column rather than writing a ragged row", refused)

# ---- 4. two same-seed runs identical ----------------------------------------------
print("\n       --- reproducibility ---")
s_a, _, _ = run("repro_1", seed=7, max_epochs=3)
s_b, _, _ = run("repro_2", seed=7, max_epochs=3)
h_a = checkpoint_hash(s_a["checkpoint_final_path"])
h_b = checkpoint_hash(s_b["checkpoint_final_path"])
wa = load_checkpoint(s_a["checkpoint_final_path"])["model_state"]
wb = load_checkpoint(s_b["checkpoint_final_path"])["model_state"]
same_w = all(torch.equal(wa[k], wb[k]) for k in wa)
print(f"       run1 best={s_a['best_metric_value']}  run2 best={s_b['best_metric_value']}")
check("TWO SAME-SEED RUNS produce identical weights", same_w)
check("two same-seed runs produce identical best metric",
      s_a["best_metric_value"] == s_b["best_metric_value"])
check("two same-seed runs produce identical final train loss",
      s_a["train_loss_final"] == s_b["train_loss_final"])

s_c, _, _ = run("repro_3", seed=9, max_epochs=3)
wc = load_checkpoint(s_c["checkpoint_final_path"])["model_state"]
check("a DIFFERENT seed produces different weights (seeds are real, not cosmetic)",
      not all(torch.equal(wa[k], wc[k]) for k in wa))

# ---- 5. resume identical -----------------------------------------------------------
print("\n       --- resume ---")
straight, _, _ = run("resume_straight", seed=5, max_epochs=4)
part, _, _ = run("resume_part", seed=5, max_epochs=2)
resumed, _, _ = run("resume_part", seed=5, max_epochs=4,
                    resume_from=part["checkpoint_final_path"])
w_straight = load_checkpoint(straight["checkpoint_final_path"])["model_state"]
w_resumed = load_checkpoint(resumed["checkpoint_final_path"])["model_state"]
maxdiff = max(float((w_straight[k].float() - w_resumed[k].float()).abs().max()) for k in w_straight)
print(f"       max |straight-through - resumed| weight diff = {maxdiff:.3e}")
check(f"RESUME IDENTICAL: 2+2 epochs == 4 epochs straight through (maxdiff={maxdiff:.2e})",
      maxdiff < 1e-6)

# ---- 5b. train/val Boundary-F1 gap monitoring (Phase 6-RUN instruction) -----------
# Training set is small (~170 images after the val carve-out); the professor/operator
# instruction was to watch the train/val Boundary F1 gap every epoch and flag it in run notes
# if it exceeds 0.15 "at convergence". Two things are checked: (a) real wiring -- a
# train_eval_loader actually produces train_* columns and gap fields through a genuine
# training run; (b) the flagging ARITHMETIC and MESSAGE, forced deterministically by
# monkeypatching the two evaluation calls, since a toy 4-image run cannot reliably reproduce a
# real generalisation gap on demand.
print("\n       --- train/val Boundary F1 gap monitoring ---")


def gap_cfg(max_epochs=1, threshold=0.15):
    cfg = tiny_cfg(max_epochs)
    cfg["training"]["track_train_val_gap"] = True
    cfg["training"]["overfit_gap_threshold"] = threshold
    return cfg


# Regression guard for the bug this section's development just found: collect_rows=False must
# still produce a REAL summary, not an empty one from summarise([]).
res_norows = validate(tiny_model(10), loader, device="cpu", collect_rows=False)
check("collect_rows=False still returns a non-empty summary (not summarise([]))",
      bool(res_norows["summary"]) and "val_boundary_f1_2px" in res_norows["summary"])
check("collect_rows=False returns an EMPTY rows list (that part is still skipped)",
      res_norows["rows"] == [])
res_rows = validate(tiny_model(10), loader, device="cpu", collect_rows=True)
check("collect_rows=True and collect_rows=False AGREE on the summary "
      "(collect_rows must never change the computed values, only row detail)",
      abs(res_norows["summary"]["val_boundary_f1_2px"]
          - res_rows["summary"]["val_boundary_f1_2px"]) < 1e-9)

# (a) wiring: a real short run with a genuine (if tiny) train_eval_loader
train_eval_loader = TinyLoader(items[:2], batch_size=2)
cfg_wire = gap_cfg(max_epochs=2)
t_wire = Trainer(model=tiny_model(11), loss_fn=build_loss_from_config(cfg_wire),
                 train_loader=loader, val_loader=loader, config=cfg_wire,
                 run_id="gap_wiring", device="cpu", output_root=tmp,
                 config_fingerprint="wiring", train_eval_loader=train_eval_loader,
                 log_tensorboard=False)
check("Trainer.track_train_val_gap is True when config enables it AND a loader is supplied",
      t_wire.track_train_val_gap is True)
summary_wire = t_wire.fit()
wire_rows = list(csv.DictReader(Path(summary_wire["csv_log"]).open(newline="", encoding="utf-8")))
check("epochs.csv carries a train_val_gap column when tracking is enabled",
      all("train_val_gap" in r for r in wire_rows))
check("epochs.csv carries at least one train_* metric column (from the train subsample)",
      any(k.startswith("train_boundary_f1") for k in wire_rows[0]))
check("run summary carries train_val_gap_final/at_best_epoch/overfit_gap_threshold keys",
      {"train_val_gap_final", "train_val_gap_at_best_epoch", "overfit_gap_threshold"}
      <= set(summary_wire))
check("overfit_gap_threshold in the summary matches the config (0.15)",
      float(summary_wire["overfit_gap_threshold"]) == 0.15)

# (b) a run WITHOUT a train_eval_loader must not track the gap, and must report "" not NaN
t_notrack = Trainer(model=tiny_model(11), loss_fn=build_loss_from_config(tiny_cfg(1)),
                    train_loader=loader, val_loader=loader, config=tiny_cfg(1),
                    run_id="gap_notrack", device="cpu", output_root=tmp,
                    config_fingerprint="notrack", log_tensorboard=False)
check("no train_eval_loader -> tracking is off even though the base config enables it "
      "elsewhere", t_notrack.track_train_val_gap is False)
summary_notrack = t_notrack.fit()
check("gap fields report '' (not NaN) when tracking never ran",
      summary_notrack["train_val_gap_final"] == ""
      and summary_notrack["overfit_gap_threshold"] == "")

# (c) deterministic flagging: force a LARGE gap and confirm the WARNING fires
cfg_big = gap_cfg(max_epochs=1, threshold=0.15)
t_big = Trainer(model=tiny_model(12), loss_fn=build_loss_from_config(cfg_big),
                train_loader=loader, val_loader=loader, config=cfg_big,
                run_id="gap_big", device="cpu", output_root=tmp,
                config_fingerprint="big", train_eval_loader=train_eval_loader,
                log_tensorboard=False)
t_big.evaluate_train_subsample = lambda epoch: {
    "summary": {"train_boundary_f1_2px": 0.95}, "rows": []}
t_big.validate = lambda epoch: {
    "summary": {"val_boundary_f1_2px": 0.10, "val_loss": 0.5}, "rows": []}
summary_big = t_big.fit()
print(f"       forced-gap case: gap={summary_big['train_val_gap_final']:.4f}  "
      f"notes={summary_big['notes']!r}")
check("a forced 0.85 gap (0.95 train vs 0.10 val) is computed correctly",
      abs(float(summary_big["train_val_gap_final"]) - 0.85) < 1e-9)
check("a gap exceeding the threshold triggers a WARNING in run notes",
      "WARNING" in summary_big["notes"] and "overfitting" in summary_big["notes"])
check("the warning names the actual gap value and the threshold",
      "0.8500" in summary_big["notes"] and "0.15" in summary_big["notes"])

# (d) deterministic flagging: a SMALL gap must NOT be flagged
cfg_small = gap_cfg(max_epochs=1, threshold=0.15)
t_small = Trainer(model=tiny_model(13), loss_fn=build_loss_from_config(cfg_small),
                  train_loader=loader, val_loader=loader, config=cfg_small,
                  run_id="gap_small", device="cpu", output_root=tmp,
                  config_fingerprint="small", train_eval_loader=train_eval_loader,
                  log_tensorboard=False)
t_small.evaluate_train_subsample = lambda epoch: {
    "summary": {"train_boundary_f1_2px": 0.50}, "rows": []}
t_small.validate = lambda epoch: {
    "summary": {"val_boundary_f1_2px": 0.48, "val_loss": 0.5}, "rows": []}
summary_small = t_small.fit()
check("a gap (0.02) under the threshold produces NO warning",
      summary_small["notes"] == "")
check("a small gap is still recorded numerically even though it wasn't flagged",
      abs(float(summary_small["train_val_gap_final"]) - 0.02) < 1e-9)

# ---- 6. checkpoint-mask guard, tracker, validator ---------------------------------
print("\n       --- guards ---")
try:
    load_checkpoint(best, expected_channel_mask_hash="deadbeef")
    guarded = False
except ValueError:
    guarded = True
check("load_checkpoint REFUSES a channel-mask hash mismatch (§9: different networks)", guarded)

tr = BestCheckpointTracker(monitor="val_boundary_f1_2px", mode="max", patience=3)
for e, v in enumerate([0.1, 0.2, 0.15, 0.15, 0.15]):
    tr.update(e, v)
check("BestCheckpointTracker finds the best epoch", tr.best_epoch == 1)
check("BestCheckpointTracker triggers plateau stop after patience", tr.should_stop())
tr2 = BestCheckpointTracker(patience=2)
tr2.update(0, float("nan"))
check("NaN never counts as an improvement", tr2.best_value is None)
check("tracker state survives a round-trip (resume keeps its patience counter)",
      (lambda t: (t.load_state_dict(tr.state_dict()), t.best_epoch)[1])(
          BestCheckpointTracker()) == 1)

m = tiny_model(3)
res = validate(m, loader, device="cpu")
check(f"validator returns ONE ROW PER IMAGE ({len(res['rows'])} rows for {N} images)",
      len(res["rows"]) == N)
check("per-image rows carry image_id (needed to pair the Wilcoxon tests)",
      all(r.get("image_id") for r in res["rows"]))
check("per-image rows carry all three frozen metrics",
      all({"dice", "boundary_f1_2px", "boundary_f1_5px", "hd95"} <= set(r)
          for r in res["rows"]))
check("summary exposes the monitored metric", "val_boundary_f1_2px" in res["summary"])

# ---- 7. BatchNorm recalibration ---------------------------------------------------
print("\n       --- BatchNorm recalibration (master §9) ---")
m = tiny_model(4)
bns = [x for x in m.modules() if isinstance(x, torch.nn.BatchNorm2d)]
for b in bns:
    b.running_mean.fill_(5.0)
recal = recalibrate_batchnorm(m, list(loader), num_batches=2, device="cpu")
check(f"recalibration touched all {recal['n_bn_layers']} BatchNorm layers",
      recal["n_bn_layers"] == len(bns))
check("stale running statistics were reset and re-estimated",
      float(bns[0].running_mean.abs().mean()) < 5.0)
check("recalibration records a batch-order fingerprint (proves B/B'/C matched)",
      bool(recal.get("batch_order_fingerprint")))
w_before = {k: v.clone() for k, v in m.state_dict().items() if "running" not in k
            and "num_batches" not in k}
recalibrate_batchnorm(m, list(loader), num_batches=2, device="cpu")
check("recalibration does NOT change any weight (frozen forward passes only)",
      all(torch.equal(w_before[k], m.state_dict()[k]) for k in w_before))
r1 = {"B": recal, "C": dict(recal)}
assert_identical_recalibration(r1)
try:
    assert_identical_recalibration({"B": recal, "C": {**recal,
                                                      "batch_order_fingerprint": "x"}})
    caught = False
except AssertionError:
    caught = True
check("assert_identical_recalibration RAISES on divergent streams", caught)

# ---- 8. the runner ----------------------------------------------------------------
print("\n       --- experiments/multi_seed_train.py ---")
r = subprocess.run([sys.executable, str(ROOT / "experiments" / "multi_seed_train.py"),
                    "--model-config", "configs/model_a_dcr.yaml",
                    "--seed-index", "1", "--dry-run"],
                   capture_output=True, text=True, cwd=ROOT)
print("       " + "\n       ".join(
    [ln for ln in r.stdout.splitlines() if "curvature proxy" in ln or "seeds" in ln]))
check(f"runner --dry-run composes and validates a config (exit {r.returncode})",
      r.returncode == 0)
check("runner echoes the sigma it resolved FROM THE YAML",
      "smooth_sigma=2.0" in r.stdout and "from YAML, not hardcoded" in r.stdout)
check("runner selects the requested seed index", "'model_init_seed': 2" in r.stdout)
runner_src = (ROOT / "experiments" / "multi_seed_train.py").read_text(encoding="utf-8")
check("runner has no hardcoded hyperparameter literals for sigma/lr/batch",
      "curvature_smooth_sigma =" not in runner_src and "lr =" not in runner_src)

# ---- 9. cross-environment portability ----------------------------------------------
print("\n       --- portability across requirements-local / requirements-kaggle ---")
for f in ("requirements-local.txt", "requirements-kaggle.txt"):
    check(f"{f} exists", (ROOT / f).is_file())
def code_only(path: Path) -> str:
    """Source with comments and docstrings stripped.

    Matching raw text would flag the very prose that documents WHY an API is avoided --
    the same false-positive class as the Phase 3 out-of-scope scan. ast.unparse drops
    comments outright, and docstrings are removed explicitly.
    """
    import ast as _ast
    tree = _ast.parse(path.read_text(encoding="utf-8"))
    for node in _ast.walk(tree):
        if isinstance(node, (_ast.Module, _ast.ClassDef, _ast.FunctionDef,
                             _ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, _ast.Expr) and isinstance(first.value, _ast.Constant)                     and isinstance(first.value.value, str):
                node.body = node.body[1:] or [_ast.Pass()]
    return _ast.unparse(tree)


trainer_code = code_only(ROOT / "src" / "training" / "trainer.py")
check("trainer CODE does not pass verbose= to ReduceLROnPlateau (removed in newer torch)",
      "verbose=" not in trainer_code)
check("trainer CODE does not touch torch.cuda.amp / torch.amp on the default path "
      "(mixed precision OFF, master §10)",
      "torch.cuda.amp" not in trainer_code and "torch.amp" not in trainer_code)
check("trainer still DOCUMENTS why those APIs are avoided (prose kept, code clean)",
      "verbose=" in (ROOT / "src" / "training" / "trainer.py").read_text(encoding="utf-8"))
ck_code = code_only(ROOT / "src" / "training" / "checkpointing.py")
check("checkpointing passes weights_only explicitly (stable across torch 2.1-2.6+)",
      "weights_only=False" in ck_code)

shutil.rmtree(tmp, ignore_errors=True)

print()
print("=" * 78)
if failures:
    print(f"RESULT: FAIL ({len(failures)} check(s) failed)")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("RESULT: PASS — Phase 6 verification complete.")
print("=" * 78)
print("Smoke test used 2x1x64x64 synthetic tensors only; real 1024px training is Kaggle-only.")
