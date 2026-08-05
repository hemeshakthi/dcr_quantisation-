# Authoritative Source Review

Executed per PROJECT_MASTER.md §0 (Authoritative Reference Policy) and §0a. Search date: 2026-08-06.
Search order followed: (1) published paper, (2) supplementary, (3) official author repo, (4) official
implementation, (5) project page, (6) author GitHub, (7) institutional repo, (8) dataset links cited by
the paper, (9) references cited inside the paper, (10) corrigendum/updates.

---

## 1. Primary paper — existence and access status

- **Confirmed to exist.** Asha, S.B. & Gopakumar, G. "Discriminative Curvature Regularization Loss for
  Boundary Segmentation in Microscopy Cell Images." *SN Computer Science* 7(1):64.
  DOI: `10.1007/s42979-025-04633-7`. Published by Springer, 07 Jan 2026. [PAPER]
- Verified via Semantic Scholar Graph API (`paperId a8e330faa9e192ccd1471bf338fe9bb33c1f7ffe`,
  DBLP key `journals/sncs/AshaG26`): authors S. B. Asha and G. Gopakumar, affiliation Amrita School of
  Computing, Amrita Vishwa Vidyapeetham, India. [PAPER]
- **Verified CLOSED ACCESS via Unpaywall API** (`api.unpaywall.org/v2/10.1007/s42979-025-04633-7`):
  `is_oa: false`, `oa_status: "closed"`, `best_oa_location: null`, `has_repository_copy: false`. No
  green-OA copy, no institutional-repository mirror, no author preprint indexed anywhere. [OFFICIAL —
  confirms paywall status, not paper content]
- Semantic Scholar's own record has `abstract: null`, `references: null` with an explicit publisher
  note: *"The following paper fields have been elided by the publisher."* Even third-party indexers
  cannot surface the abstract or reference list programmatically. [OFFICIAL]
- `link.springer.com/article/10.1007/s42979-025-04633-7` redirects (302/303) through Springer's IDP
  cookie-authorization flow and back to the same paywalled article page — full text, and the numbered
  reference list, are NOT accessible through this session's tooling. [OFFICIAL]
- **Conclusion: the paper body is confirmed paywalled with no legitimate free-access route found.**
  This matches the master file's §0a expectation exactly. The abstract-level claims already recorded
  in master §0a (curvature embedding space, +9.18%/+6.21% Dice improvements, touching cells/low-contrast
  focus, "minimal computational overhead") are corroborated by multiple independent search-result
  snippets quoting the same abstract text, so they are treated as reliable [PAPER] facts even though
  this session could not re-fetch the abstract as a single verbatim block.
- The numbered reference list (refs 5, 6, 9, 10, 15, 17–19, 27, 29, 30, 33, 35–44) recorded in master
  §0a could **not** be independently re-verified in this session (paywalled). It is carried forward
  from the master file as-is and should be treated as provisional until the PDF is obtained.

## 2. Official author repo / implementation / project page

- **No official GitHub repository, project page, or code release found** for this paper or its dCR loss
  formulation, despite targeted searches (`github.com` + author names, `github.com` + "dCR loss" /
  "curvature regularization loss" + segmentation, arXiv preprint search for "curvature embedding space"
  + boundary segmentation). [OFFICIAL — negative result]
- No arXiv preprint of this paper exists under any variant of the title searched. [OFFICIAL — negative
  result]
- Author faculty page (`amrita.edu/faculty/gopakumarg/`) lists G. Gopakumar's publications and confirms
  a PhD student, **Asha S. Benny**, working on "Deep learning framework for automatic cell segmentation
  in microscopy images" — consistent with S.B. Asha being this same person and this dCR paper being
  part of that PhD lineage. The faculty page provides no code/dataset/PDF links. [OFFICIAL]
- ORCID for Gopakumar: `0000-0003-1559-5150` (not separately mined for additional links; low expected
  yield beyond what's already found).
- **Conclusion: §8's DCRLoss surrogate remains an [ASSUMPTION]** until the professor supplies the exact
  equation or code. No official implementation exists publicly to fall back on.

## 3. Dataset identity — new corroborating evidence found

The master file (§5) already flagged the two Zenodo StarDist/TrackMate datasets (refs 37/38) as the
highest-confidence candidates because they are the only datasets named in the paper's public reference
list. This session found **additional circumstantial evidence strengthening that inference**, without
being able to confirm it against the dCR paper's own (paywalled) body:

- The same author group (S.B. Asha, G. Gopakumar, plus G.R.K.S. Subrahmanyam) published a direct
  predecessor paper: **"Saliency and ballness driven deep learning framework for cell segmentation in
  bright field microscopic images"** (SBU-net), *Engineering Applications of Artificial Intelligence*,
  vol. 118, art. 105704, 2023. That paper states it was evaluated on **"two publicly available
  bright-field datasets of T cells and pancreatic cancer cells"** with 5-fold cross-validation, mean IoU
  0.804/0.829, mean Dice 0.891/0.906. [LITERATURE — same-lab predecessor work, not the dCR paper itself]
- Cross-checked the two Zenodo candidates and confirmed their descriptions match this predecessor
  paper's dataset description exactly:
  - **Zenodo 4034929** — "Combining StarDist and TrackMate example 2 – T cell dataset" (Roy & Jacquemet,
    2020). 209 paired brightfield images + masks. T cells migrating on ICAM-1, tracked with
    StarDist/TrackMate via ZeroCostDL4Mic. → matches "T cells." [LITERATURE]
  - **Zenodo 4034939** — "Combining StarDist and TrackMate example 3 – Flow chamber dataset" (Jacquemet,
    2020). **AsPC1 pancreatic cancer cells** flowing in a microfluidic chamber, brightfield, tracked with
    StarDist/TrackMate via ZeroCostDL4Mic. → matches "pancreatic cancer cells" exactly. [LITERATURE]
- **This raises confidence that the dCR paper (same lab, same PhD lineage, same problem framing) reuses
  these same two datasets**, but this is still an inference from a predecessor paper, not a citation
  found inside the dCR paper's own accessible text. It remains logged as an [ASSUMPTION] in
  `implementation_assumptions.md` (upgraded rationale, NOT upgraded to [PAPER]) until the professor or
  the PDF confirms the dCR paper's actual data-availability statement.
- BCCD, Cellpose, DeepBacs, and generic "HeLa bright-field" remain lower-confidence fallback candidates
  per master §5 — no new evidence found either way this session.

## 4. Architecture / loss lineage — no contradictions found

- No new information changes master §7/§8's treatment of U-Net family, Dice loss, or the dCR
  surrogate. Searches for the exact dCR formulation, any GitHub implementation, or a public preprint
  all returned negative results (see §2 above). [OFFICIAL — negative result, reconfirms master's
  existing [ASSUMPTION] status for the exact equation]

## 5. Open questions that MUST go to the professor before Phase 2 finalizes

1. Can you provide the PDF (or paywall-free access) to the full paper body?
2. What are the exact dataset(s) and their source links/DOIs used in the dCR paper? (Working hypothesis:
   Zenodo 4034929 T-cell + Zenodo 4034939 AsPC1 flow-chamber, based on predecessor-paper reuse pattern —
   please confirm or correct.)
3. What is the exact train/val/test split (or k-fold protocol) used?
4. What is the exact dCR loss equation / formula (the "curvature embedding space" formulation)? Is there
   associated code?
5. What image resolution, normalization, and augmentation pipeline was used?
6. What optimizer, learning rate (+ schedule), weight decay, batch size, epoch count/stopping criterion,
   and weight-initialization scheme were used?
7. What post-processing (if any) was applied to model outputs before metric computation?
8. Does the paper specify exact U-Net channel widths, or is the "standard" [64,128,256,512,1024]
   configuration in master §7 the correct assumption?
9. Is there any existing official/unofficial code for dCR loss or SBU-net that the professor is aware of
   but that isn't publicly indexed?

## 6. Summary table

| Item | Status | Label |
|---|---|---|
| Paper existence, DOI, authors, venue | Confirmed | [PAPER] |
| Abstract-level claims (curvature embedding space, +9.18%/+6.21% Dice) | Confirmed via multiple snippets | [PAPER] |
| Full paper body / methods / equations | Paywalled, inaccessible | — (blocks [PAPER] confirmation) |
| Numbered reference list (refs 5,6,9,10,15,17-19,27,29,30,33,35-44) | Carried from master, not re-verifiable this session | [PAPER] (provisional) |
| Official GitHub / code repo | Searched, not found | [OFFICIAL] (negative) |
| arXiv preprint | Searched, not found | [OFFICIAL] (negative) |
| Dataset identity (T-cell + AsPC1 flow-chamber) | Strong circumstantial match via predecessor paper | [ASSUMPTION] (elevated confidence, still unconfirmed) |
| Exact dCR equation | Not recoverable | [ASSUMPTION] (per master §8) |
| Preprocessing / hyperparameters | Not recoverable | [ASSUMPTION] (per master §0a/§10) |

**Action required:** all 9 open questions in §5 must go to the professor before Phase 2 (training
pipeline) finalizes, per master §0a and §5.
