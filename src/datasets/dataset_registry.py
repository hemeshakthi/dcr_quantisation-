"""Central dataset registry for the dCR compression study.

Single source of truth for dataset identity, provenance, license, and CONFIDENCE LABEL.
Shared by download_datasets.py, quality_check.py, and make_split.py so that the three
never disagree about what a dataset is or how confident we are that the paper used it.

SOURCE LABELS (PROJECT_MASTER.md §0):
  [PAPER]      reproduced from the dCR paper
  [OFFICIAL]   from author repo/supplementary
  [LITERATURE] supported by recent peer-reviewed work
  [ENGINEERING]best practice for reproducibility
  [ASSUMPTION] not recoverable from any authoritative source

CRITICAL (master §5, §0a): the exact dataset identities used by the dCR paper live in the
PAYWALLED paper body. The ONLY datasets named in the paper's PUBLIC reference list are the
two StarDist/TrackMate Zenodo records (ref 37 = 4034939, ref 38 = 4034929). Everything else
is [ASSUMPTION]. Nothing here may be treated as confirmed until the professor or the PDF
confirms it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_SPLITS = ROOT / "data" / "splits"

# Characters illegal in Windows filenames. Python's zipfile sanitises these to "_" when
# extracting on Windows but NOT on Linux, which would make image_ids diverge between a
# local Windows run and a Kaggle (Linux) run. We apply the same mapping on EVERY platform
# so that image_ids -- and therefore the split CSVs -- are byte-identical everywhere.
# [ENGINEERING] cross-platform reproducibility (master §10 determinism intent).
_ILLEGAL_WIN_CHARS = ':<>|"?*'
_SANITISE_TABLE = str.maketrans(_ILLEGAL_WIN_CHARS, "_" * len(_ILLEGAL_WIN_CHARS))


def canonical_name(name: str) -> str:
    """Platform-independent filename canonicalisation. See note above."""
    return name.translate(_SANITISE_TABLE)


@dataclass
class DatasetSpec:
    name: str                      # short key, used as data/raw/<name>/
    title: str
    confidence: str                # source label + rationale
    acquisition: str               # AUTO (scriptable) | MANUAL (human must fetch)
    license: str
    source_url: str
    doi: Optional[str] = None
    zenodo_record: Optional[int] = None
    expected_archive: Optional[str] = None
    expected_bytes: Optional[int] = None
    # Relative paths INSIDE the extracted archive holding paired images/masks.
    # Verified by reading the real ZIP central directory (not guessed).
    pair_dirs: list[tuple[str, str]] = field(default_factory=list)
    citation: str = ""
    notes: str = ""
    manual_instructions: str = ""

    @property
    def raw_dir(self) -> Path:
        return DATA_RAW / self.name

    @property
    def meta_path(self) -> Path:
        return self.raw_dir / "_meta.json"


# --------------------------------------------------------------------------------------
# HIGHEST-CONFIDENCE CANDIDATES
# The only datasets named in the dCR paper's PUBLIC reference list (refs 37/38).
# Still [ASSUMPTION] as to actual USE by the dCR paper -- the body is paywalled and could
# not be read. Confidence is elevated (not confirmed) by the predecessor-paper evidence
# recorded in docs/source_review.md §3.
# --------------------------------------------------------------------------------------
TCELL = DatasetSpec(
    name="stardist_tcell",
    title="Combining StarDist and TrackMate example 2 - T cell dataset",
    confidence=(
        "[PAPER ref 38 -- named in the paper's public reference list] + [ASSUMPTION] that the "
        "paper actually TRAINED on it. Elevated by predecessor SBU-net (EAAI 2023, same authors) "
        "reporting a 'T cells' bright-field dataset. NOT CONFIRMED -- see source_review.md §3."
    ),
    acquisition="AUTO",
    license="CC-BY-4.0",
    source_url="https://zenodo.org/records/4034929",
    doi="10.5281/zenodo.4034929",
    zenodo_record=4034929,
    expected_archive="T cell dataset.zip",
    expected_bytes=331_431_787,
    pair_dirs=[
        ("T cell dataset/Training dataset/Training_source",
         "T cell dataset/Training dataset/Training_target"),
        ("T cell dataset/Quality control dataset/Training_source",
         "T cell dataset/Quality control dataset/Training_target"),
    ],
    citation="Roy, N.H. & Jacquemet, G. (2020). Zenodo. doi:10.5281/zenodo.4034929",
    notes=(
        "VERIFIED by reading the remote ZIP central directory: 209 paired image/mask files under "
        "'Training dataset' (Training_source/Training_target), plus 1 pair under 'Quality control "
        "dataset'. Brightfield .tif, 1024x1024, 645 nm pixel size, 10x phase-contrast objective, "
        "Zeiss Axiovert 200M, frames every 30 s over 10 min. "
        "'Test Data/' holds a SINGLE 57 MB UNLABELLED time-lapse stack (no mask) and is therefore "
        "EXCLUDED from this study -- see implementation_assumptions.md A12. "
        "TIME-LAPSE: frames are temporally correlated -> grouped splitting is mandatory (A11)."
    ),
)

FLOWCHAMBER = DatasetSpec(
    name="stardist_flowchamber",
    title="Combining StarDist and TrackMate example 3 - Flow chamber dataset",
    confidence=(
        "[PAPER ref 37 -- named in the paper's public reference list] + [ASSUMPTION] that the "
        "paper actually TRAINED on it. Elevated by predecessor SBU-net (EAAI 2023, same authors) "
        "reporting a 'pancreatic cancer cells' bright-field dataset; this record is AsPC1 "
        "pancreatic cancer cells. NOT CONFIRMED -- see source_review.md §3."
    ),
    acquisition="AUTO",
    license="CC-BY-4.0",
    source_url="https://zenodo.org/records/4034939",
    doi="10.5281/zenodo.4034939",
    zenodo_record=4034939,
    expected_archive="Flow chamber dataset.zip",
    expected_bytes=359_391_863,
    pair_dirs=[
        ("Flow chamber dataset/Training dataset/Training_source",
         "Flow chamber dataset/Training dataset/Training_target"),
    ],
    citation="Follain, G. & Jacquemet, G. (2020). Zenodo. doi:10.5281/zenodo.4034939",
    notes=(
        "VERIFIED by reading the remote ZIP central directory: ~57 paired image/mask files under "
        "'Training dataset'. AsPC1 pancreatic cancer cells perfused at 300 um/s in a microfluidic "
        "channel; Zeiss Laser-TIRF 3, 10X objective; .tif, 1024x1024, 650 nm pixel size. "
        "'Test data/' and 'Quality check dataset/' are handled separately -- see A12. "
        "TIME-LAPSE: filenames encode timepoints from the SAME field of view "
        "(e.g. 't:1-5941 - FOV1 B.czi #1.tif', 't:301-...', 't:601-...'), so adjacent frames are "
        "near-duplicates -> grouped splitting is mandatory (A11)."
    ),
)

# --------------------------------------------------------------------------------------
# LOWER-CONFIDENCE FALLBACK CANDIDATES -- ALL [ASSUMPTION], MANUAL ONLY.
# These are NOT named in the paper's public reference list. They are listed in master §5
# as candidates consistent with the authors' prior work. None may be used without the
# professor confirming it. download_datasets.py will NOT fetch these automatically.
# --------------------------------------------------------------------------------------
BCCD = DatasetSpec(
    name="bccd",
    title="BCCD (Blood Cell Count and Detection)",
    confidence=(
        "[ASSUMPTION] NOT named in the paper's public reference list. Listed in master §5 only "
        "because it matches the authors' blood-cell lineage (ref 6). Requires professor confirmation."
    ),
    acquisition="MANUAL",
    license="MIT",
    source_url="https://github.com/Shenggan/BCCD_Dataset",
    citation="Shenggan, BCCD_Dataset, GitHub (MIT).",
    notes=(
        "BLOCKING ISSUE: BCCD ships BOUNDING-BOX annotations (PASCAL VOC XML), NOT segmentation "
        "masks. Per master §5 we must NOT fabricate masks from boxes. The segmentation-mask source "
        "must be confirmed with the professor before this dataset can be used at all."
    ),
    manual_instructions=(
        "1. git clone https://github.com/Shenggan/BCCD_Dataset\n"
        "2. Place under data/raw/bccd/\n"
        "3. DO NOT PROCEED: confirm the segmentation-mask source with the professor first.\n"
        "   BCCD's own annotations are bounding boxes and cannot yield boundary-accurate masks."
    ),
)

CELLPOSE = DatasetSpec(
    name="cellpose_cyto",
    title="Cellpose cyto dataset",
    confidence="[ASSUMPTION] NOT named in the paper's public reference list. Requires professor confirmation.",
    acquisition="MANUAL",
    license="Non-commercial (see cellpose.org terms) -- REVIEW BEFORE USE",
    source_url="https://www.cellpose.org/dataset",
    citation="Stringer, C. et al. Cellpose. Nat Methods (2021).",
    notes="Non-commercial license may be incompatible with publication/redistribution requirements. Review before use.",
    manual_instructions=(
        "1. Visit https://www.cellpose.org/dataset and accept the license terms.\n"
        "2. Download manually and place under data/raw/cellpose_cyto/\n"
        "3. Confirm with the professor that the dCR paper used this dataset before using it."
    ),
)

DEEPBACS = DatasetSpec(
    name="deepbacs_ecoli",
    title="DeepBacs E. coli segmentation dataset",
    confidence="[ASSUMPTION] NOT named in the paper's public reference list. Requires professor confirmation.",
    acquisition="MANUAL",
    license="CC-BY-4.0",
    source_url="https://zenodo.org/communities/deepbacs",
    citation="Spahn, C. et al. DeepBacs. Commun Biol (2022).",
    notes="Bacterial (rod-shaped) morphology differs substantially from the mammalian cells in the paper's focus.",
    manual_instructions=(
        "1. Browse https://zenodo.org/communities/deepbacs and select the specific record.\n"
        "2. Place under data/raw/deepbacs_ecoli/\n"
        "3. Confirm the exact record with the professor first."
    ),
)

HELA = DatasetSpec(
    name="hela_brightfield",
    title="HeLa bright-field (source unconfirmed)",
    confidence="[ASSUMPTION] Source repository not even identified. Requires professor confirmation.",
    acquisition="MANUAL",
    license="Dataset-dependent -- UNKNOWN until the source is identified",
    source_url="(unidentified -- master §5 lists this as 'confirm repo')",
    citation="(unidentified)",
    notes="Master §5 itself marks the repository as unconfirmed. Cannot be actioned without the professor.",
    manual_instructions="Cannot be actioned: the source repository itself is unidentified. Ask the professor.",
)

REGISTRY: dict[str, DatasetSpec] = {
    d.name: d for d in (TCELL, FLOWCHAMBER, BCCD, CELLPOSE, DEEPBACS, HELA)
}

# Datasets the pipeline treats as the working set unless the professor says otherwise.
PRIMARY_CANDIDATES = [TCELL.name, FLOWCHAMBER.name]

PROFESSOR_ACTION_BANNER = r"""
================================================================================
  ACTION REQUIRED -- PROFESSOR CONFIRMATION OUTSTANDING (master §0a, §5)
================================================================================
  The dCR paper body is PAYWALLED (verified closed access: Unpaywall is_oa=false,
  no repository copy; Semantic Scholar abstract/references elided by publisher).
  The EXACT dataset identities used by the paper CANNOT be recovered from any
  public source.

  The ONLY datasets named in the paper's PUBLIC reference list are:
      ref 37 -> Zenodo 4034939  (StarDist/TrackMate flow chamber, AsPC1 pancreatic)
      ref 38 -> Zenodo 4034929  (StarDist/TrackMate T cell)
  These are treated as HIGHEST-CONFIDENCE CANDIDATES ONLY. That the paper actually
  TRAINED on them remains an [ASSUMPTION] (see implementation_assumptions.md A01).

  ALL other candidates (BCCD, Cellpose cyto, DeepBacs, HeLa bright-field) are
  [ASSUMPTION] with NO supporting citation in the public record and are NOT
  downloaded automatically by this script.

  >>> PLEASE ASK THE PROFESSOR TO CONFIRM, IN WRITING:
      1. The EXACT list of datasets used in the dCR paper.
      2. The EXACT download links / DOIs for each.
      3. Which subset (training/test/quality-control) of each was used.
      4. The exact train/val/test split or k-fold protocol.
      5. Whether any dataset outside this candidate list was used.

  Until then, every result produced by this pipeline is conditional on an
  UNCONFIRMED dataset choice and must be reported as such.
================================================================================
"""

BCCD_MASK_TODO_BANNER = r"""
********************************************************************************
  TODO -- BLOCKING -- BCCD SEGMENTATION MASK SOURCE UNRESOLVED
********************************************************************************
  BCCD (github.com/Shenggan/BCCD_Dataset) ships BOUNDING-BOX annotations in
  PASCAL VOC XML format. It does NOT ship segmentation masks.

  PROJECT_MASTER.md §5 is explicit:
      "BCCD ships as bounding boxes, not masks -- do NOT fabricate masks from
       boxes; confirm mask source with professor."

  Deriving masks from bounding boxes would produce rectangles, which is
  scientifically INVALID for this study: the entire research question concerns
  BOUNDARY GEOMETRY and CURVATURE. Rectangular pseudo-masks have no meaningful
  curvature and would silently destroy the curvature stratification (master §11),
  the boundary metrics (master §6), and the failure taxonomy (master §12).

  >>> ASK THE PROFESSOR: if BCCD is genuinely part of the study, what is the
      SOURCE of its segmentation masks? Without a real mask source, BCCD MUST
      remain excluded.

  THIS PIPELINE WILL NOT GENERATE BCCD MASKS. NO FABRICATION. (Golden Rule 1.)
********************************************************************************
"""


def print_professor_banner() -> None:
    print(PROFESSOR_ACTION_BANNER)


def print_bccd_todo() -> None:
    print(BCCD_MASK_TODO_BANNER)
