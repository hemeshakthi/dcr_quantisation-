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

Kaggle installs `requirements-kaggle.txt` (master §16 pins). The source tree runs unmodified
under both that and `requirements-local.txt` — verified by `audit/verify_phase6.py`, which
checks the specific API-drift hazards: no `verbose=` on `ReduceLROnPlateau`, no `torch.amp`
usage on the default path, explicit `weights_only` on `torch.load`.

Do not `pip install` anything that changes a pinned version mid-notebook.

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
