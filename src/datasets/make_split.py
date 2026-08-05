"""Phase 1 step 4: build the fixed 70/30 train/test split with a hard leakage guarantee.

Split ratio 70/30 is FROZEN by PROJECT_MASTER.md §0b [ENGINEERING -- professor-set].

WHY THIS IS NOT A RANDOM PER-IMAGE SPLIT
----------------------------------------
Both StarDist/TrackMate datasets are TIME-LAPSE movies: consecutive frames show the same
cells moments apart. A random per-image split would put frame t in train and frame t+1 in
test, which is textbook leakage and would inflate every metric -- fatally, for a study whose
entire claim is about fine boundary geometry.

The split is therefore made over ATOMIC GROUPS, where a group is a connected component of:

    (near-duplicate graph from quality_check.py)  UNION  (filename series groups)

Any two images that are perceptually near-identical, or that come from the same acquisition
series, are forced into the same side of the split. This satisfies master §5's "split by
slide if found" and makes the required leakage check true BY CONSTRUCTION -- it is then
re-verified independently at the end.

This is a WRITE-ONCE artefact. Re-running with --force overwrites and re-logs.

Usage:
    python src/datasets/make_split.py
    python src/datasets/make_split.py --seed 42 --force
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.datasets.dataset_registry import (  # noqa: E402
    DATA_PROCESSED,
    DATA_SPLITS,
    REGISTRY,
    PRIMARY_CANDIDATES,
    ROOT,
    print_professor_banner,
)

# [ENGINEERING] master §0b. FROZEN.
TRAIN_FRACTION = 0.70
# [ASSUMPTION A15] split seed. 42 matches the seed master §13 fixes for visual-gate
# pre-selection; using one project-wide constant keeps the data pipeline auditable.
DEFAULT_SEED = 42


class DisjointSet:
    def __init__(self, items):
        self.parent = {i: i for i in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def build_groups(df: pd.DataFrame, dup_info: dict) -> dict[str, str]:
    """Union near-duplicate edges and series groups into atomic split units."""
    ids = df["image_id"].tolist()
    ds = DisjointSet(ids)

    idset = set(ids)
    for edge in dup_info.get("near_duplicate_pairs", []):
        if edge["a"] in idset and edge["b"] in idset:
            ds.union(edge["a"], edge["b"])

    for _, sub in df.groupby("series_key"):
        members = sub["image_id"].tolist()
        for other in members[1:]:
            ds.union(members[0], other)

    return {i: ds.find(i) for i in ids}


def _greedy_assign(keys: list[str], group_sizes: dict[str, int],
                   target_train: float) -> tuple[set[str], set[str]]:
    """Largest-first greedy: place each group wherever it keeps train closest to target."""
    train, test = set(), set()
    n_train = 0
    for k in keys:
        s = group_sizes[k]
        if abs((n_train + s) - target_train) <= abs(n_train - target_train):
            train.add(k); n_train += s
        else:
            test.add(k)
    return train, test


def split_groups(group_sizes: dict[str, int], seed: int,
                 strata: dict[str, str] | None = None) -> tuple[set[str], set[str]]:
    """Assign whole groups to train/test, hitting TRAIN_FRACTION by image count.

    Groups are shuffled with a fixed seed then sorted largest-first, so a few very large
    groups cannot blow the ratio apart. If `strata` is given, the 70/30 target is applied
    WITHIN each stratum (e.g. per acquisition condition) so that both sides contain every
    condition -- at the cost of hitting the global ratio less precisely.
    """
    rng = np.random.default_rng(seed)

    def ordered(keys: list[str]) -> list[str]:
        ks = sorted(keys)                 # deterministic base order
        rng.shuffle(ks)                   # seeded shuffle
        ks.sort(key=lambda k: -group_sizes[k])   # largest first (stable)
        return ks

    train: set[str] = set()
    test: set[str] = set()

    if strata:
        buckets: dict[str, list[str]] = {}
        for k in group_sizes:
            buckets.setdefault(strata.get(k, "_"), []).append(k)
        for _, keys in sorted(buckets.items()):
            sub_total = sum(group_sizes[k] for k in keys)
            tr, te = _greedy_assign(ordered(keys), group_sizes, TRAIN_FRACTION * sub_total)
            train |= tr; test |= te
    else:
        total = sum(group_sizes.values())
        train, test = _greedy_assign(ordered(list(group_sizes)), group_sizes,
                                     TRAIN_FRACTION * total)

    # guard: never produce an empty side
    if not test and train:
        smallest = min(train, key=lambda k: group_sizes[k])
        train.remove(smallest); test.add(smallest)
    if not train and test:
        largest = max(test, key=lambda k: group_sizes[k])
        test.remove(largest); train.add(largest)
    return train, test


def derive_condition(series_key: str) -> str:
    """Coarse acquisition condition, used only for composition reporting/stratification.

    'T cells 1' -> 'T cells';  'VCAM 2-2' -> 'VCAM';  'FOV1 B.czi #' -> 'FOV'
    """
    import re
    s = re.split(r"[\d]", series_key, maxsplit=1)[0]
    return s.strip(" -_") or series_key


def verify_no_leakage(df: pd.DataFrame, dup_info: dict) -> list[str]:
    """Independent re-verification: no image, near-duplicate, or series spans the split."""
    violations = []
    assign = dict(zip(df["image_id"], df["split"]))

    ids = df["image_id"].tolist()
    if len(ids) != len(set(ids)):
        violations.append("duplicate image_id present in split table")

    for edge in dup_info.get("near_duplicate_pairs", []):
        a, b = edge["a"], edge["b"]
        if a in assign and b in assign and assign[a] != assign[b]:
            violations.append(
                f"NEAR-DUPLICATE ACROSS SPLIT (hamming={edge['hamming']}): "
                f"{a} [{assign[a]}] <-> {b} [{assign[b]}]")

    for key, sub in df.groupby("series_key"):
        sides = set(sub["split"])
        if len(sides) > 1:
            violations.append(f"SERIES ACROSS SPLIT: series='{key}' spans {sorted(sides)}")

    return violations


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--force", action="store_true", help="overwrite an existing split")
    ap.add_argument("--stratify", action=argparse.BooleanOptionalAction, default=True,
                    help="apply the 70/30 target WITHIN each acquisition condition so both "
                         "sides contain every condition (DEFAULT). --no-stratify applies it "
                         "per dataset only, which hits 70/30 more exactly but can leave a "
                         "whole condition absent from one side. See A17.")
    args = ap.parse_args()

    print_professor_banner()

    keys = args.datasets or list(PRIMARY_CANDIDATES)
    train_csv = DATA_SPLITS / "train.csv"
    test_csv = DATA_SPLITS / "test.csv"
    if (train_csv.exists() or test_csv.exists()) and not args.force:
        print(f"Split already exists at {DATA_SPLITS} -- WRITE-ONCE guard.")
        print("Re-run with --force to overwrite (this invalidates any trained model).")
        return 1

    frames, dupinfo_all = [], {}
    for key in keys:
        inv = DATA_PROCESSED / f"{key}_inventory.csv"
        dup = DATA_PROCESSED / f"{key}_nearduplicates.json"
        if not inv.exists():
            print(f"  SKIP {key}: no inventory. Run quality_check.py first.")
            continue
        df = pd.read_csv(inv)
        info = json.loads(dup.read_text(encoding="utf-8")) if dup.exists() else {}
        groups = build_groups(df, info)
        df["group_id"] = df["image_id"].map(groups)
        frames.append(df)
        dupinfo_all[key] = info
        print(f"  {key}: {len(df)} images -> {df['group_id'].nunique()} atomic groups")

    if not frames:
        print("\nRESULT: PARTIAL -- no inventories found. Run:")
        print("  python src/datasets/download_datasets.py")
        print("  python src/datasets/quality_check.py")
        return 1

    all_df = pd.concat(frames, ignore_index=True)

    # Group ids are namespaced per dataset already (image_id carries the dataset prefix),
    # so groups never merge across datasets.
    all_df["condition"] = all_df["dataset"] + "/" + all_df["series_key"].map(derive_condition)
    sizes = all_df.groupby("group_id").size().to_dict()

    # The 70/30 target is ALWAYS applied per dataset. Pooling both datasets into a single
    # global assignment lets one dataset's ratio drift arbitrarily to compensate for the
    # other's (observed: 85% train for one, 30% for the other, while the pooled figure
    # still read ~70%). Each dataset is its own corpus and gets its own 70/30.
    strat_col = "condition" if args.stratify else "dataset"
    strata = all_df.groupby("group_id")[strat_col].agg(lambda s: sorted(set(s))[0]).to_dict()
    print(f"\n  stratifying the 70/30 target by: {strat_col}")
    train_g, test_g = split_groups(sizes, args.seed, strata)
    all_df["split"] = np.where(all_df["group_id"].isin(train_g), "train", "test")

    # ---- leakage verification -----------------------------------------------------
    violations = []
    for key in keys:
        sub = all_df[all_df["dataset"] == key]
        if len(sub):
            violations += verify_no_leakage(sub, dupinfo_all.get(key, {}))

    n_tr = int((all_df["split"] == "train").sum())
    n_te = int((all_df["split"] == "test").sum())
    total = n_tr + n_te

    print("\n" + "=" * 78)
    print("SPLIT RESULT")
    print("=" * 78)
    print(f"  seed                 : {args.seed}")
    print(f"  target train fraction: {TRAIN_FRACTION:.2f}  (FROZEN, master §0b)")
    print(f"  achieved             : train={n_tr} ({n_tr/total:.3f})  test={n_te} ({n_te/total:.3f})")
    print(f"  atomic groups        : train={len(train_g)}  test={len(test_g)}")
    for key in keys:
        sub = all_df[all_df["dataset"] == key]
        if len(sub):
            a = int((sub['split'] == 'train').sum()); b = int((sub['split'] == 'test').sum())
            print(f"    {key:<24} train={a:<5} test={b:<5} ({a/(a+b):.3f} train)")

    print("\n  ---------------- COMPOSITION BY CONDITION ----------------")
    comp = all_df.pivot_table(index="condition", columns="split", values="image_id",
                              aggfunc="count", fill_value=0)
    for cond, row in comp.iterrows():
        a = int(row.get("train", 0)); b = int(row.get("test", 0))
        flag = ""
        if a == 0:
            flag = "  <-- ABSENT FROM TRAIN"
        elif b == 0:
            flag = "  <-- ABSENT FROM TEST"
        print(f"    {cond:<34} train={a:<5} test={b:<5}{flag}")
    missing = [c for c, r in comp.iterrows()
               if int(r.get("train", 0)) == 0 or int(r.get("test", 0)) == 0]
    if missing:
        print("\n    WARNING [A17]: the condition(s) above appear on only ONE side of the")
        print("    split. This is a consequence of holding the FROZEN 70/30 ratio (master")
        print("    §0b) while keeping acquisition series atomic (A11) -- the T-cell subset")
        print("    contains only 2 movies, so no group-atomic assignment can hit 70/30")
        print("    within it. Re-run with --stratify to balance composition instead, at the")
        print("    cost of the global ratio. THIS IS A DECISION FOR THE PROFESSOR.")

    print("\n  ---------------- LEAKAGE CHECK ----------------")
    if violations:
        print(f"  FAILED: {len(violations)} violation(s)")
        for v in violations[:25]:
            print(f"    ! {v}")
        return 1
    print("  PASSED: no image, near-duplicate pair, or acquisition series spans train/test.")

    # ---- write --------------------------------------------------------------------
    DATA_SPLITS.mkdir(parents=True, exist_ok=True)
    cols = ["image_id", "dataset", "subset", "image_path", "mask_path",
            "series_key", "group_id", "n_objects", "fg_fraction", "img_h", "img_w"]
    cols = [c for c in cols if c in all_df.columns]
    all_df[all_df["split"] == "train"][cols].to_csv(train_csv, index=False)
    all_df[all_df["split"] == "test"][cols].to_csv(test_csv, index=False)

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": args.seed,
        "train_fraction_target": TRAIN_FRACTION,
        "train_fraction_achieved": round(n_tr / total, 6),
        "n_train": n_tr, "n_test": n_te, "n_total": total,
        "datasets": keys,
        "grouping_rule": "connected components of (near-duplicate graph UNION series groups)",
        "stratified_by": strat_col,
        "train_fraction_deviation": round(n_tr / total - TRAIN_FRACTION, 6),
        "per_dataset": {
            k: {
                "train": int(((all_df["dataset"] == k) & (all_df["split"] == "train")).sum()),
                "test": int(((all_df["dataset"] == k) & (all_df["split"] == "test")).sum()),
            } for k in keys
        },
        "conditions_on_one_side_only": missing,
        "leakage_check": "PASSED",
        "source_labels": {
            "70/30 ratio": "[ENGINEERING] master §0b, professor-set",
            "grouped splitting": "[ENGINEERING] required by leakage check; see A11",
            "seed": "[ASSUMPTION] A15",
            "dataset identity": "[ASSUMPTION] A01 -- UNCONFIRMED",
        },
        "warning": "Dataset identity is UNCONFIRMED (A01). If the professor names different "
                   "datasets, this split must be regenerated and all models retrained.",
    }
    (DATA_SPLITS / "split_manifest.json").write_text(json.dumps(manifest, indent=2),
                                                     encoding="utf-8")
    print(f"\n  wrote {train_csv.relative_to(ROOT)}  ({n_tr} rows)")
    print(f"  wrote {test_csv.relative_to(ROOT)}  ({n_te} rows)")
    print(f"  wrote {(DATA_SPLITS / 'split_manifest.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
