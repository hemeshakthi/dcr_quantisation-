"""Single-job training runner -- the ONE entry point Kaggle calls (master §10, §17).

One invocation trains ONE (model, ratio, seed) combination and appends ONE experiment-log row.
Master §10 requires the three seeds to be logged separately, so the runner is deliberately
single-job: a Kaggle notebook loops over seeds by calling this repeatedly, and a crashed seed
never takes the other two with it.

    python experiments/multi_seed_train.py --model-config configs/model_a_dcr.yaml \
        --seed-index 0 --device cuda --output-root /kaggle/working

*** CONFIG VALUES COME FROM THE YAMLs, NEVER FROM A NOTEBOOK CELL. ***
Every hyperparameter is composed here from configs/ (see src/config.py). A Kaggle notebook that
restated a value in a cell could drift from the repo, and `curvature_smooth_sigma` in particular
would silently reintroduce the pixelation bug Phase 5 fixed -- at sigma=0 the curvature proxy
scores a corner 0.3166 against a straight edge 0.3333, i.e. it stops discriminating entirely.
`validate_config` rejects a dCR run with a missing or non-positive sigma, the resolved value is
echoed at startup, and both it and the config fingerprint are written into the experiment-log
row, so a drift is detectable after the fact as well as prevented up front.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import (compose_run_config, config_fingerprint,  # noqa: E402
                        validate_config)
from src.losses.combined import build_loss_from_config  # noqa: E402
from src.models.unet import UNet  # noqa: E402
from src.training.experiment_log import append_run, build_run_record  # noqa: E402
from src.training.seed_manager import apply_run_determinism, seeds_for_run  # noqa: E402
from src.training.trainer import Trainer  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-config", required=True,
                    help="configs/model_*.yaml -- the ONLY place hyperparameters come from")
    ap.add_argument("--shared-config", default="training_shared.yaml")
    ap.add_argument("--pruning-config", default="pruning.yaml")
    ap.add_argument("--seed-index", type=int, default=0, choices=(0, 1, 2),
                    help="which of the 3 model_init_seeds (master §10)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output-root", default=None,
                    help="/kaggle/working on Kaggle; repo root locally")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--resume-from", default=None)
    ap.add_argument("--channel-mask-hash", default=None,
                    help="required for B/B'/C so all three provably share one network (§9)")
    ap.add_argument("--max-epochs", type=int, default=None,
                    help="override for smoke tests ONLY; never for a real run")
    ap.add_argument("--executed-on", default="kaggle")
    ap.add_argument("--dry-run", action="store_true",
                    help="compose, validate and print the config; train nothing")
    args = ap.parse_args(argv)

    cfg = compose_run_config(args.model_config, args.shared_config, args.pruning_config)
    if args.max_epochs is not None:
        cfg.setdefault("training", {})["max_epochs"] = int(args.max_epochs)

    problems = validate_config(cfg)
    if problems:
        print("CONFIG VALIDATION FAILED:")
        for p in problems:
            print(f"  ! {p}")
        return 2

    fp = config_fingerprint(cfg)
    seeds = seeds_for_run(cfg, args.seed_index)
    model_name = cfg.get("run", {}).get("name", Path(args.model_config).stem)
    run_id = args.run_id or f"{model_name}_seed{seeds.model_init_seed}"
    boundary = (cfg.get("loss", {}) or {}).get("boundary", {}) or {}

    print("=" * 74)
    print(f"RUN {run_id}")
    print("=" * 74)
    print(f"  model config      : {args.model_config}")
    print(f"  config fingerprint: {fp[:16]}...")
    print(f"  seeds             : {seeds.as_dict()}")
    print(f"  optimizer         : {cfg['optimizer']['name']} lr={cfg['optimizer']['lr']} "
          f"wd={cfg['optimizer']['weight_decay']}")
    print(f"  batch/epochs      : {cfg['training']['batch_size']} / "
          f"{cfg['training']['max_epochs']}")
    print(f"  mixed precision   : {cfg['training']['mixed_precision']}")
    print(f"  monitor           : {cfg['early_stopping']['monitor']} "
          f"(patience {cfg['early_stopping']['patience']})")
    # Echoed so the Kaggle log shows the value ACTUALLY used, resolved from the YAML.
    print(f"  boundary loss     : {boundary.get('name') or 'none'} "
          f"beta={boundary.get('beta')} band_width={boundary.get('band_width')}")
    print(f"  curvature proxy   : {boundary.get('curvature_proxy')} "
          f"smooth_sigma={boundary.get('curvature_smooth_sigma')}  <- from YAML, not hardcoded")

    if args.dry_run:
        print("\n[dry-run] configuration valid; nothing trained.")
        return 0

    det = apply_run_determinism(cfg, seeds)
    print(f"  determinism       : {json.dumps({k: det[k] for k in
          ('cudnn_deterministic', 'cudnn_benchmark', 'use_deterministic_algorithms')})}")

    from src.datasets.dataset import build_dataloaders
    train_loader, val_loader, train_eval_loader = build_dataloaders(
        cfg, seeds.augmentation_seed, seeds.data_order_seed)
    print(f"  train/val images  : {len(train_loader.dataset)} / {len(val_loader.dataset)}")
    if train_eval_loader is not None:
        print(f"  train-eval subsample (overfit monitor): "
              f"{len(train_eval_loader.dataset)} images, "
              f"gap threshold {cfg['training'].get('overfit_gap_threshold')}")

    model = UNet.from_config(cfg, init_seed=seeds.model_init_seed)
    loss_fn = build_loss_from_config(cfg)

    trainer = Trainer(model=model, loss_fn=loss_fn, train_loader=train_loader,
                      val_loader=val_loader, config=cfg, run_id=run_id,
                      device=args.device, output_root=args.output_root,
                      config_fingerprint=fp,
                      channel_mask_hash=args.channel_mask_hash,
                      resume_from=args.resume_from,
                      train_eval_loader=train_eval_loader)
    summary = trainer.fit()

    record = build_run_record(run_id, cfg, summary, config_fingerprint=fp,
                              seeds=seeds.as_dict(), executed_on=args.executed_on)
    out_root = Path(args.output_root) if args.output_root else ROOT
    log_path = out_root / "results" / "tables" / "experiment_log.csv"
    append_run(record, log_path)

    print(f"\n  best {summary['best_metric_name']} = {summary['best_metric_value']} "
          f"at epoch {summary['best_epoch']}")
    print(f"  status: {summary['status']}   duration: {summary['duration_minutes']} min")
    if summary.get("notes"):
        print(f"  {summary['notes']}")
    print(f"  experiment_log row -> {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
