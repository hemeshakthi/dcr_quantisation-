"""Phase 1 step 1: acquire (or document) the candidate datasets into data/raw/<name>/.

For AUTO datasets (the two Zenodo StarDist/TrackMate records named in the paper's public
reference list) this resolves the file list from the Zenodo REST API at runtime -- URLs and
checksums are NEVER hardcoded from memory -- downloads with resume, verifies the MD5
reported by the API, extracts, and writes data/raw/<name>/_meta.json.

For MANUAL datasets (every lower-confidence [ASSUMPTION] candidate) it prints exact manual
instructions and writes a _meta.json stub with status="NOT_ACQUIRED" so that nothing
downstream can mistake an unconfirmed dataset for an available one.

Usage:
    python src/datasets/download_datasets.py                 # primary candidates only
    python src/datasets/download_datasets.py --all           # include manual instructions
    python src/datasets/download_datasets.py --datasets stardist_tcell
    python src/datasets/download_datasets.py --dry-run       # document, download nothing
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.datasets.dataset_registry import (  # noqa: E402
    REGISTRY,
    PRIMARY_CANDIDATES,
    DatasetSpec,
    canonical_name,
    print_bccd_todo,
    print_professor_banner,
)

ZENODO_API = "https://zenodo.org/api/records/{record}"
# 256 KiB. Deliberately modest: this project has been run on a host whose pagefile is
# effectively disabled (commit limit == physical RAM), where larger streaming buffers
# raised MemoryError mid-download. Smaller chunks + resume make the download survivable.
CHUNK = 1 << 18
MAX_ATTEMPTS = 5


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def fetch_zenodo_record(record: int) -> dict:
    """Resolve the live Zenodo record. Source of truth for URLs, sizes, checksums."""
    r = requests.get(ZENODO_API.format(record=record), timeout=60)
    r.raise_for_status()
    return r.json()


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def download_with_resume(url: str, dest: Path, expected_size: int | None = None) -> None:
    """Stream a download, resuming a partial file if one exists."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    existing = dest.stat().st_size if dest.exists() else 0
    if expected_size and existing == expected_size:
        print(f"    already complete ({existing:,} bytes), skipping download")
        return
    headers = {}
    mode = "wb"
    if existing:
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"
        print(f"    resuming from {existing:,} bytes")

    with requests.get(url, headers=headers, stream=True, timeout=(30, 300)) as r:
        if existing and r.status_code == 200:
            # Server ignored the Range header -> restart cleanly rather than corrupt.
            print("    server ignored Range; restarting download from 0")
            existing, mode = 0, "wb"
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0)) + existing
        done = existing
        last_pct = -1
        with open(dest, mode) as f:
            for chunk in r.iter_content(CHUNK):
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = int(done * 100 / total)
                    if pct != last_pct and pct % 5 == 0:
                        print(f"    {pct:3d}%  ({done:,}/{total:,} bytes)", flush=True)
                        last_pct = pct
    print(f"    downloaded {dest.stat().st_size:,} bytes")


def safe_extract(archive: Path, dest: Path) -> list[str]:
    """Extract a zip, rejecting path traversal, canonicalising names identically on all OSes.

    Python's zipfile silently rewrites Windows-illegal characters (':' etc.) on Windows but
    not on Linux. We apply the same mapping everywhere so that a local Windows run and a
    Kaggle (Linux) run produce byte-identical paths -- and therefore identical image_ids.
    """
    dest.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            rel = canonical_name(info.filename)
            target = (dest / rel).resolve()
            if not str(target).startswith(str(dest.resolve())):
                raise RuntimeError(f"unsafe path in archive: {info.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out, CHUNK)
            written.append(rel)
    return written


def write_meta(spec: DatasetSpec, status: str, extra: dict | None = None) -> None:
    meta = {
        "name": spec.name,
        "title": spec.title,
        "status": status,
        "confidence_label": spec.confidence,
        "acquisition": spec.acquisition,
        "license": spec.license,
        "source_url": spec.source_url,
        "doi": spec.doi,
        "citation": spec.citation,
        "notes": spec.notes,
        "recorded_at_utc": _utc_now(),
        "professor_confirmation_required": True,
        "professor_confirmation_received": False,
        "master_reference": "PROJECT_MASTER.md §5; docs/source_review.md §3; "
                            "docs/implementation_assumptions.md A01",
    }
    if extra:
        meta.update(extra)
    spec.raw_dir.mkdir(parents=True, exist_ok=True)
    spec.meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"    wrote {spec.meta_path.relative_to(Path(__file__).resolve().parents[2])}")


def acquire_auto(spec: DatasetSpec, dry_run: bool) -> bool:
    print(f"\n=== {spec.name} :: {spec.title} ===")
    print(f"  confidence: {spec.confidence}")
    print(f"  license   : {spec.license}")
    print(f"  source    : {spec.source_url}")

    try:
        rec = fetch_zenodo_record(spec.zenodo_record)
    except Exception as e:
        print(f"  !! could not reach Zenodo API: {type(e).__name__}: {e}")
        write_meta(spec, "NOT_ACQUIRED", {"error": f"{type(e).__name__}: {e}"})
        return False

    files = rec.get("files", [])
    print(f"  Zenodo reports {len(files)} file(s):")
    file_records = []
    for f in files:
        key = f.get("key")
        size = f.get("size")
        checksum = f.get("checksum", "")
        print(f"    - {key}  ({size:,} bytes)  checksum={checksum}")
        file_records.append({"key": key, "size": size, "checksum": checksum})

    if dry_run:
        print("  [dry-run] not downloading")
        write_meta(spec, "NOT_ACQUIRED", {"files": file_records, "dry_run": True})
        return False

    archive_dir = spec.raw_dir / "_archive"
    extracted_ok = True
    for f in files:
        key, size = f.get("key"), f.get("size")
        url = (f.get("links", {}) or {}).get("self") or \
              f"https://zenodo.org/api/records/{spec.zenodo_record}/files/{key}/content"
        dest = archive_dir / key
        print(f"  downloading {key} ...")
        ok = False
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                download_with_resume(url, dest, expected_size=size)
                ok = True
                break
            except Exception as e:
                # Resume means a retry continues from wherever the previous attempt died,
                # so transient MemoryError/network drops do not restart from zero.
                print(f"  !! attempt {attempt}/{MAX_ATTEMPTS} failed: "
                      f"{type(e).__name__}: {e}")
                if attempt < MAX_ATTEMPTS:
                    print("     retrying (resuming from partial file) ...")
        if not ok:
            print(f"  !! download failed after {MAX_ATTEMPTS} attempts")
            extracted_ok = False
            continue

        # Verify against the checksum the API itself reported (format "md5:<hex>").
        checksum = f.get("checksum", "")
        if checksum.startswith("md5:"):
            print("    verifying md5 ...")
            got = _md5(dest)
            if got != checksum.split(":", 1)[1]:
                print(f"  !! MD5 MISMATCH: expected {checksum}, got md5:{got}")
                extracted_ok = False
                continue
            print("    md5 OK")

        if key.lower().endswith(".zip"):
            print(f"    extracting into {spec.raw_dir} ...")
            names = safe_extract(dest, spec.raw_dir)
            print(f"    extracted {len(names)} files")

    write_meta(
        spec,
        "ACQUIRED" if extracted_ok else "PARTIAL",
        {
            "files": file_records,
            "zenodo_record": spec.zenodo_record,
            "zenodo_doi": rec.get("doi"),
            "zenodo_title": rec.get("metadata", {}).get("title"),
            "extracted_into": str(spec.raw_dir),
            "pair_dirs": [list(p) for p in spec.pair_dirs],
        },
    )
    return extracted_ok


def acquire_manual(spec: DatasetSpec) -> None:
    print(f"\n=== {spec.name} :: {spec.title} ===")
    print(f"  confidence: {spec.confidence}")
    print(f"  license   : {spec.license}")
    print(f"  source    : {spec.source_url}")
    print("  ACQUISITION: MANUAL -- this script will NOT download it automatically,")
    print("               because it is an [ASSUMPTION] with no citation in the paper's")
    print("               public reference list.")
    if spec.notes:
        print(f"  NOTES: {spec.notes}")
    print("  MANUAL INSTRUCTIONS:")
    for line in spec.manual_instructions.splitlines():
        print(f"    {line}")
    if spec.name == "bccd":
        print_bccd_todo()
    write_meta(spec, "NOT_ACQUIRED", {"manual_instructions": spec.manual_instructions})


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="dataset keys to process (default: primary candidates)")
    ap.add_argument("--all", action="store_true",
                    help="process every registry entry, including MANUAL ones")
    ap.add_argument("--dry-run", action="store_true", help="document only, download nothing")
    args = ap.parse_args()

    print_professor_banner()

    if args.datasets:
        keys = args.datasets
    elif args.all:
        keys = list(REGISTRY)
    else:
        keys = list(PRIMARY_CANDIDATES)

    unknown = [k for k in keys if k not in REGISTRY]
    if unknown:
        print(f"ERROR: unknown dataset key(s): {unknown}")
        print(f"       known keys: {list(REGISTRY)}")
        return 2

    acquired, documented = [], []
    for key in keys:
        spec = REGISTRY[key]
        if spec.acquisition == "AUTO":
            ok = acquire_auto(spec, args.dry_run)
            (acquired if ok else documented).append(key)
        else:
            acquire_manual(spec)
            documented.append(key)

    # BCCD's blocking mask issue must be visible even when BCCD wasn't selected.
    if "bccd" not in keys:
        print_bccd_todo()

    print("\n================ DOWNLOAD SUMMARY ================")
    print(f"  acquired            : {acquired or '(none)'}")
    print(f"  documented-only     : {documented or '(none)'}")
    print("  Every dataset above remains an [ASSUMPTION] as to whether the dCR paper")
    print("  actually used it. See docs/implementation_assumptions.md A01.")
    print("=================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
