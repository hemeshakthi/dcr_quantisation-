# Backend API Contract — v2 (implemented)

**Supersedes the v1 draft** (5 endpoints: health/models/images/case/selfcheck), written when this
was still a forward-looking spec for an unbuilt backend. The backend-authoring phase gave an
explicit, more specific 3-endpoint shape (`GET /images`, `POST /infer`, `GET /health`) and asked
for **server-rendered** overlays and error maps, superseding v1's client-side error-map decision.
This document describes what `backend/` actually implements; `frontend/` was updated to match.

Master §14, Golden Rules 1–2 still govern every endpoint below:
- **LIVE inference on real checkpoints. No placeholders/hardcoded/fake/mock**, anywhere.
- **Two identical model outputs is a BUG**, never an acceptable response.
- If C doesn't beat B, the API returns the real numbers regardless.

---

## 0. Scope decisions

### A35 — Which checkpoint per model (unchanged from v1)
Every model column loads `checkpoints/<run_name>_seed1/best.pt` (seed 1, best-on-val-Boundary-F1
per A31). `run_name` ∈ `{vanilla, A_dcr, B_pruned, B_prime_generic, C_pruned_dcr}`. B/B′/C are at
the primary ratio 0.5 (master §14). See `docs/implementation_assumptions.md` A35.

### A39 — Error maps and contour overlays are now SERVER-rendered, not client-side
v1 (frontend phase) decided error maps would be computed client-side from two fetched masks, to
avoid a second network round trip. **This backend phase's instruction explicitly lists "error
maps base64" as part of `/infer`'s response**, overriding that decision. The backend now uses
`src/visualization/contours.py::draw_contour` (line-style overlay) and `::error_map` (solid
FP / hatched FN) — both implemented for real in this phase — so the exact same B&W rendering
logic is available to Phase 13's static paper figures and to this live API. The frontend was
simplified accordingly: it now displays three ready-made `<img>` elements per model
(prediction / overlay / error map) instead of doing canvas pixel comparison and SVG polyline
construction itself. See A39 in `docs/implementation_assumptions.md`.

### Lazy, cached model loading
Models are loaded from disk **lazily on first request**, not blocking backend startup — with a
CPU-only host and no checkpoints necessarily present at boot, blocking startup on 5 model loads
(or worse, on a load that will simply fail because Kaggle hasn't run yet) would make the server
unable to even answer `/api/health`. Each successfully loaded model is cached in the
`ModelRegistry` for the process lifetime. The self-check probe (§1 below) runs once, lazily, on
the **first** `/api/health` call (which triggers loading whatever checkpoints exist), and its
result is cached from then on — "startup self-check" is read as "compute once, not per request",
matching v1's reasoning; it just cannot literally block process startup on a CPU host where a
checkpoint may not even exist yet.

### Degraded mode is not fabrication
If a checkpoint is missing or fails to load, that model is excluded from `models_loaded` and
listed in `models_unavailable` with the real reason. `/api/infer` returns results only for
models that are actually loaded (never a fabricated entry for a missing one). This is the
project's current REAL state (`checkpoints/` is empty until Kaggle Phase 6-RUN and Phase 9
pruning both run) — the backend must report that honestly, not synthesize a fifth column.

---

## 1. `GET /api/health`

Triggers lazy loading of every expected checkpoint (if not already cached) and runs the
collapse probe once. Merges v1's `/api/health` and `/api/selfcheck` into one call, and also
carries the static per-model metadata the frontend needs to build its UI shell (v1's
`/api/models` is dropped as a separate endpoint — its content is folded in here, since building
it requires the same loaded-model state health already computes).

```jsonc
{
  "status": "ok",                          // "ok" (all 5 loaded, no collapse) | "degraded"
  "models_loaded": ["vanilla", "A"],        // whichever of the 5 actually loaded
  "models_unavailable": ["B", "B_prime", "C"],
  "unavailable_reasons": {
    "B": "checkpoint not found at checkpoints/B_pruned_seed1/best.pt"
  },
  "models": [                               // static metadata, ONLY for models_loaded
    {
      "model_id": "vanilla", "label": "Vanilla", "is_reference": false,
      "line_style": "dotted", "pruning_ratio": null,
      "params_count": 31036481, "model_size_bytes": 124345872,
      "checkpoint_hash": "8f2a...c3", "checkpoint_path": "checkpoints/vanilla_seed1/best.pt",
      "seed": 1
    }
    // ...
  ],
  "collapse_check": {
    "probe_image_id": "...",
    "collapsed": false,
    "pairs": [
      {"model_a": "vanilla", "model_b": "A", "identical": false, "mean_abs_diff": 0.1342}
      // all C(n,2) pairs among models_loaded; empty list if fewer than 2 models loaded
    ]
  },
  "test_images_available": 98
}
```

`collapse_check.collapsed = true` is what the frontend's red-free "MODEL COLLAPSE DETECTED"
banner (A36) keys off. Comparison is on **pre-threshold sigmoid probabilities**
(`np.array_equal`, atol 0), matching `configs/evaluation.yaml`'s
`identical_output_check.compare_pre_threshold: true` — unchanged from v1.

---

## 2. `GET /api/images`

Unchanged from v1. Test-split only, labelled by curvature bin.

```jsonc
{
  "default_image_id": "...",   // a HIGH-bin image, deterministic (seed 42, master §13 convention)
  "images": [
    {"image_id": "...", "dataset": "stardist_tcell", "curvature_bin": "high", "curvature_score": 0.2134}
    // every image in data/splits/test.csv
  ]
}
```

---

## 3. `POST /api/infer`

Request body:
```json
{ "image_id": "stardist_tcell/.../foo.tif" }
```

The primary endpoint. Runs live inference for every currently-loaded model on the requested
image (cached per `(image_id, model_id)` after the first real forward pass — the cache stores a
genuine computed result, it is never pre-seeded or fabricated) and returns everything needed to
render one full comparison screen.

**Every value in the example response below is an ILLUSTRATIVE PLACEHOLDER showing the JSON
SHAPE only** (this contract was written before any model was trained) **-- not a real measured
result.** The live values are computed for real by `src/metrics/segmentation.py` once real
checkpoints exist; see `audit/verify_backend_integration.py` for genuinely computed (if tiny/
synthetic-fixture) example numbers, and `docs/synthetic_validation_manifest.md` for the full
inventory of every synthetic artifact in this repo.

```jsonc
{
  "image_id": "...",
  "dataset": "stardist_tcell",
  "curvature_bin": "high",
  "curvature_score": 0.2134,
  "cell_size": 331.3,
  "contrast": 0.42,
  "original_png_base64": "iVBORw0KG...",
  "ground_truth": {
    "mask_png_base64": "iVBORw0KG...",
    "overlay_png_base64": "iVBORw0KG..."   // original + GT contour (style "solid"), server-rendered
  },
  "models": [
    {
      "model_id": "vanilla",
      "prediction_mask_png_base64": "iVBORw0KG...",
      "overlay_png_base64": "iVBORw0KG...",     // original + this model's contour (its own line_style)
      "error_map_png_base64": "iVBORw0KG...",   // FP solid / FN hatched vs ground truth, B&W
      "metrics": {
        "dice": 0.8123, "boundary_f1_2px": 0.5511, "boundary_f1_5px": 0.7820,
        "hd95": 12.34                            // JSON null if undefined (either mask empty)
      },
      "model_size_bytes": 124345872,
      "params_count": 31036481,
      "pruning_ratio": null
    }
    // ... one entry per model in models_loaded (§1) at call time
  ]
}
```

**Metrics use the exact functions in `src/metrics/segmentation.py`** (`dice_score`,
`boundary_f1`, `hd95`) — the same code path as the study's own per-image tables, so a number
shown here and a number in a results table can never silently disagree.

**Overlays and error maps use `src/visualization/contours.py`** (`draw_contour`, `error_map`),
implemented for real this phase — see A39. `curvature_bin` is echoed per top-level response
(image property); `pruning_ratio`/`model_size_bytes`/`params_count` are per-model (echoed from
`/api/health`'s static metadata, so a client that only ever calls `/infer` still has them).

---

## 4. Error responses

Unknown `image_id`: HTTP 404, `{"error": "image_id '...' not found in the test split"}`.
Inference failure: HTTP 500, `{"error": "..."}`. Never a 200 with a fabricated/default body.

---

## 5. What the backend must NOT do

- Must not hardcode a static JSON fixture "for now" — every response is a live call into the
  loaded PyTorch models via `InferenceService`.
- Must not block startup waiting for checkpoints that may not exist yet (lazy load, §0).
- Must not silently substitute one model's output for another's if a checkpoint fails to load.
- Must not skip the collapse probe "to save time".
