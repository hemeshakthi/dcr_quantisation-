# Kaggle Run Contract

**Binding on PHASE 6-RUN and every later [KAGGLE] phase.** Written before the first Kaggle run.

Master §17 splits the work: authoring, verification, statistics, figures, UI and audit run in
Claude Code; only the 24 training runs and full-test inference run on Kaggle. This file states
what the Kaggle side is and is not allowed to do, because the failure modes here are silent —
a drifted value produces a *successful-looking* run with wrong results.

---

## 1. The notebook calls the runner. It does not reimplement it.

The single entry point is:

```bash
python experiments/multi_seed_train.py \
    --model-config configs/model_a_dcr.yaml \
    --seed-index 0 \
    --device cuda \
    --output-root /kaggle/working \
    --executed-on kaggle
```

A notebook cell may pass only: a config path, a seed index, a device, an output root, and
(for B/B′/C) a channel-mask hash. **Everything else comes from `configs/`.**

## 2. No hyperparameter may be restated in a notebook cell

**`curvature_smooth_sigma = 2.0` must be read from `configs/model_a_dcr.yaml` and
`configs/model_c_pruned_dcr.yaml` at training time, never hardcoded separately in any Kaggle
notebook cell. This value must never diverge between local verification and Kaggle training.**

This is not stylistic. σ became a config value in Phase 5 after measurement showed the naive
choice fails to discriminate corners at all (a corner scored 0.3166 against a straight edge's
0.3333). A cell that retyped `sigma = 1.0` — or that silently relied on a default — would
reintroduce that bug, and *both* the local and Kaggle runs would still report success. The same
rule applies to `band_width`, `beta`, `lr`, `batch_size` and every other tuned value.

Enforced in four layers, not by convention:

| Layer | Mechanism |
|---|---|
| Composition | `src.config.compose_run_config` builds every value from `configs/` |
| Front-end rejection | `validate_config` refuses a dCR run with σ missing or ≤ 0 |
| Visible echo | the runner prints the resolved σ, marked "from YAML, not hardcoded" |
| After-the-fact detection | `curvature_smooth_sigma`, `loss_band_width`, `loss_beta`, `curvature_proxy` and a `config_fingerprint` are written into every experiment-log row |

The fingerprint is the backstop: two runs that were supposed to share a configuration can be
**proven** to have done so by comparing fingerprints, rather than assumed.

## 3. Environment

Kaggle installs `requirements-kaggle.txt` (master §16 pins) — **except `torch`/`torchvision`,
which are left as Kaggle's own preinstalled, GPU-matched build (A34).** Forcing the pin would
risk silently downgrading to CPU or breaking CUDA for infrastructure reasons unrelated to the
model; the source tree is written to tolerate whatever torch version Kaggle actually provides
instead (`audit/verify_phase6.py` checks the specific API-drift hazards this depends on: no
`verbose=` on `ReduceLROnPlateau`, no `torch.amp` usage on the default path, explicit
`weights_only` on `torch.load`). The notebook prints the resolved `torch.__version__` and CUDA
availability at startup so the actual environment is recorded, not assumed.

Do not `pip install` anything else that changes a pinned version mid-notebook.

## 3a. Confirmed: σ is read from the model configs, never restated (Phase 6-RUN)

Confirmed for this run: `curvature_smooth_sigma=2.0` is read from `configs/model_a_dcr.yaml`
and `configs/model_c_pruned_dcr.yaml` at training time via `experiments/multi_seed_train.py` →
`src.config.compose_run_config`, not hardcoded separately in any Kaggle notebook cell. This value
must never diverge between local verification and Kaggle training — Cell 3 of
`experiments/kaggle_train_vanilla_a.ipynb` dry-runs both configs and prints the resolved value
before any GPU time is spent, precisely so a drift is caught before training rather than after.

## 4. What Kaggle must produce, per run

- `checkpoints/<run_id>/best.pt` and `final.pt`
- `logs/<run_id>/epochs.csv` (one row per epoch: loss, grad norm, lr, all val metrics)
- one appended row in `results/tables/experiment_log.csv`, carrying the **git commit hash**

Runs are single-job by design (master §10 logs the three seeds separately): a crashed seed must
never take the other two with it.

## 5. What must come back to Claude Code

`checkpoints/` and `logs/` and the experiment-log rows, placed at the repo paths above. Then the
[CC] verification runs: all checkpoints load, the experiment log is complete and duplicate-free,
and — critically — **Vanilla and A must produce DIFFERENT outputs on a probe image.** Identical
outputs are the model-collapse bug that killed the previous attempt at this project
(master §0b, Golden Rule 2); they are to be investigated, never accepted.

## 6. Things that are NOT Kaggle's job

Re-splitting data, regenerating `curvature_bins.csv` (write-once, locked pre-training),
re-pruning per model (B/B′/C load ONE shared mask, master §9), or computing statistics. If a
notebook appears to need any of these, the pipeline is being used wrongly.

---

## 7. Concrete instance: Phase 6-RUN (Vanilla ×3 + Model A ×3)

Notebook: `experiments/kaggle_train_vanilla_a.ipynb` (8 cells: environment check, repo/data
sanity check, config dry-run drift-guard proof, run plan, the 6-run training loop, an
experiment-log report including any overfit-gap warnings, and a download-bundle packager).

**Kaggle Dataset to create.** One dataset containing the repo AND the processed data together
(so relative paths — `configs/`, `data/processed/`, `data/splits/`, `data/curvature/` — resolve
exactly as they do locally; nothing in the code needs to change):

- Slug: **`dcr-compression-study-repo`** (must match `REPO_INPUT` in notebook Cell 2 exactly —
  edit that one line if you name it differently).
- Contents to upload: `src/`, `configs/`, `experiments/`, `requirements-kaggle.txt`,
  `data/processed/` (~1.6 GB), `data/splits/`, `data/curvature/`.
- Do **not** upload: `data/raw/` (~2 GB, not needed post-preprocessing), `checkpoints/`, `logs/`,
  `.git/` — these only inflate upload size and Kaggle's dataset storage quota.
- Simplest path: zip those directories from the repo root and use Kaggle's "New Dataset" →
  upload. If the repo is already on GitHub, Kaggle's "Add Data → GitHub" import also works, but
  `data/processed/` will still need to be added separately (GitHub import won't carry files this
  large / gitignored).

**GPU: P100.** The code is single-device (`model.to(device)`, no `DataParallel`); "GPU T4 ×2"
would only use one of the two T4s unless the trainer is extended for multi-GPU (out of scope
here), and a single P100 has higher throughput than a single T4. Use T4 ×2 (pinned to `cuda:0`)
only as a fallback if P100 quota is exhausted.

**Duration.** Master §17's own estimate is ~30–90 min per run; 6 runs sequentially is therefore
**~3–9 hours**, which can exceed a single Kaggle session depending on your tier/quota. In
practice, early stopping (patience 15 on val Boundary F1) usually cuts this well under the full
100-epoch budget — Cell 6 prints each run's actual duration and status so this is visible as it
happens. **If your session has a shorter GPU-quota window than ~9 hours, split this into two
notebook runs**: trim `RUN_PLAN` in Cell 4 to the three `model_vanilla.yaml` entries for the
first session, then to the three `model_a_dcr.yaml` entries for a second session (the
per-(model,seed) experiment-log append means partial progress is never lost between sessions).

**Files to download, and where they go locally.** From the notebook's Output pane, download
`phase6_run_artifacts.zip` (built by Cell 7) and unzip it at the repo root, so that:

```
checkpoints/vanilla_seed1/{best,final}.pt   <- from download_me/checkpoints/vanilla_seed1/
checkpoints/vanilla_seed2/{best,final}.pt
checkpoints/vanilla_seed3/{best,final}.pt
checkpoints/A_dcr_seed1/{best,final}.pt
checkpoints/A_dcr_seed2/{best,final}.pt
checkpoints/A_dcr_seed3/{best,final}.pt
logs/vanilla_seed1/epochs.csv                <- from download_me/logs/vanilla_seed1/
... (one logs/<run_id>/epochs.csv per run)
results/tables/experiment_log.csv            <- MERGE the 6 new rows into the repo's copy
                                                 (do not overwrite; the header must stay identical)
```

`checkpoints/` and `logs/` are gitignored, so placing files there is always safe. Then run:

```bash
python audit/verify_phase6_run.py
```

which checks exactly what master's VERIFICATION line asks for: all 6 checkpoints load, Vanilla
and A produce different outputs on a real test-split probe image (not merely non-crashing —
genuinely different, with the mean-absolute-difference printed for each of the 9 cross-pairs),
and the experiment log is complete. This script has been integration-tested against real (tiny)
checkpoints built the same way `Trainer.fit()` builds them, in both the passing case and a
deliberately-forced collapse case (two identical checkpoints), and correctly PASS/FAILs each.

---

## 8. Concrete instance: Phase 7-RUN (prune Model A, fine-tune B/B′/C × 2 ratios × 3 seeds)

Notebook: `experiments/kaggle_prune_and_finetune_bc.ipynb` (10 cells: markdown intro, GPU check,
writable-copy + batch_size/env setup, staged dependency install, import verification, repo/data
sanity check, prune-once-per-ratio + BN-recalibration dry run, the 18-entry RUN_PLAN, the
18-run training loop with per-run Kaggle Dataset push, and a summary/next-steps cell).
**Authored 2026-08-10, not yet executed** — it depends on Model A's real checkpoint
(`checkpoints/A_dcr_seed1/best.pt`), which does not exist locally until Phase 6-RUN's session
completes and its artifacts are downloaded back per §7 above.

**Lessons carried forward from the Phase 6-RUN session** (2026-08-10), each with its own inline
comment at the point of use in the notebook — recorded here too so a future session's notebook
inherits them even if it isn't a direct copy of this one:

| Lesson | Fix |
|---|---|
| `/kaggle/input/` is read-only; editing `configs/*.yaml` there silently no-ops | `shutil.copytree()` into `/kaggle/working/` FIRST; never symlink into `/kaggle/input/` |
| `batch_size=4` (the repo default) OOMs pruning/fine-tuning a full U-Net at 1024×1024 on a T4 | Set `batch_size=1` in the writable config copy, immediately after copying |
| `batch_size=2` was tested and is SLOWER than `1` (168.7 vs 132.1 min/run) — not just "also OOMs" | Use `1`, not `2`, even though `2` doesn't crash |
| Fragmented CUDA allocator across many sequential runs can OOM even under total-usage headroom | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| Installing all pip packages in one call intermittently produced `numpy.dtype`/`numpy.rec`/`numpy.strings` binary-incompatibility errors | Stage the install: numpy alone → verify import → scipy alone → verify import → everything else in one batch |
| Relative paths (`"experiments/multi_seed_train.py"`) made all 6 Phase 6-RUN runs fail INSTANTLY with `FileNotFoundError` | Absolute paths (`REPO_ROOT / "experiments" / ...`) AND an explicit `cwd=str(REPO_ROOT)` on every `subprocess.run()` call |
| `FileLink` / the notebook Output pane's browser download are both unreliable for files this large | `kaggle datasets create`/`kaggle datasets version` after every completed run, using Kaggle Secrets (`KAGGLE_USERNAME`/`KAGGLE_KEY`) for auth; the exact local `kaggle datasets download -d ... --unzip` command is printed after each push |
| Session GPU quota may run out mid-`RUN_PLAN` | `RUN_PLAN` is a plain editable list; a later session trims it to the not-yet-completed `(model, ratio, seed)` entries by checking `results/tables/experiment_log.csv` first (same per-run-append safety property as §7's `RUN_PLAN`) |

**GPU: T4, not P100** for this notebook specifically — the reverse of §7's Phase 6-RUN guidance
(P100 preferred there). Cell 1 asserts `"T4" in torch.cuda.get_device_name(0)` and raises
immediately if not, rather than silently training on whatever accelerator happened to be
selected and only noticing from run duration or a later OOM.

**A real gap found and fixed while authoring this notebook.** `experiments/multi_seed_train.py`
previously only ever built a model via `UNet.from_config(...)` — correct for Vanilla/A, but for
B/B′/C that would have silently trained a full-size, UNPRUNED network mislabeled as the pruned
one the first time this notebook's Cell 8 actually ran (exactly the "successful-looking run with
wrong results" failure mode this file's own opening paragraph warns against). Fixed by adding
`_build_pruned_base()` (master §9: load Model A's checkpoint, apply the shared mask, recalibrate
BatchNorm on an identical batch stream) as a real code path in the runner itself — see A42 in
`docs/implementation_assumptions.md`. Every one of B/B′/C independently redoing this (rather
than sharing one persisted "pruned base" checkpoint between jobs) keeps them single-job/
crash-isolated (§4 above) while still being provably identical, since the mask hash and the
recalibration batch order are both deterministic and verified, not merely assumed.

**Verification once all 18 rows are complete**: author and run a Phase 7-RUN verification
script mirroring `audit/verify_phase6_run.py`'s pattern (§7 above), extended for B/B′/C:
all 18 checkpoints load; B/B′/C at each ratio report IDENTICAL measured params/FLOPs and an
identical `mask_hash` (master §9); and no two of Vanilla/A/B/B′/C collapse to identical outputs
on a probe image (Golden Rule 2). This script does not exist yet — it is scoped to the future
session that actually has 18 real rows to check, matching how `verify_phase6_run.py` itself was
only written once Phase 6-RUN's real artifacts existed to verify.
