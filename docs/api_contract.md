# Backend API Contract — binding on Phase 11

**Written before the backend exists.** The frontend (this phase) is built as a strict consumer
of this contract via `fetch()`; Phase 11 implements a FastAPI backend that satisfies it exactly.
This is the same pattern as `docs/kaggle_run_contract.md`: define the interface precisely first,
so the two sides can be built independently and still agree.

Master §14, Golden Rules 1–2 govern every endpoint below:
- **LIVE inference on real checkpoints. No placeholders/hardcoded/fake/mock**, anywhere.
- **Two identical model outputs is a BUG**, never an acceptable response.
- If C doesn't beat B, the API returns the real numbers regardless — no endpoint may adjust,
  round favourably, or omit a result because it is unflattering.

---

## 0. Scope decisions this contract fixes (not yet decided by any earlier phase)

### A35 — Which checkpoint per model (which seed, which ratio) [ENGINEERING/ASSUMPTION]
Master §14 says "backend loads 5 real checkpoints (Vanilla, A, B, B′, C at **0.5**)" — the
primary pruning ratio (master §9), one checkpoint each, not all 3×2×3=18 compressed runs. Which
of the 3 seeds represents each model in the UI is otherwise unspecified. **Decision: seed 1
(`model_init_seed=1`)'s `best.pt` for every model**, i.e. `checkpoints/<run_name>_seed1/best.pt`
where `run_name` ∈ `{vanilla, A_dcr, B_pruned, B_prime_generic, C_pruned_dcr}` (matching
`configs/model_*.yaml`'s `run.name`, and `run_name + "_seed1"` matching
`experiments/multi_seed_train.py`'s run_id convention). `best.pt` (not `final.pt`) because it is
selected on val Boundary F1 (A31) — the checkpoint that best represents the model's boundary
performance, matching what the demo is trying to show. Logged in
`docs/implementation_assumptions.md` A35; **not** a statistical claim — the paper's confirmatory
numbers (Phase 8+) use all 3 seeds properly, this is a single illustrative instance for the UI.

### Images shown are TEST-split only
`GET /api/images` and `GET /api/case/{image_id}` only ever serve images from
`data/splits/test.csv`. Showing a training image would demonstrate memorisation, not
generalisation — dishonest for a tool whose entire purpose is showing real behaviour.

### Startup self-check timing
"Startup" self-check (master §14) means computed **once, at backend startup**, after all models
are loaded, and cached. `GET /api/selfcheck` returns the cached result (fast — the frontend calls
it on every page load, and re-running inference on every page load would be wasteful and is not
what "startup" implies).

### Byte-identical comparison is on PRE-THRESHOLD probabilities
Matching `configs/evaluation.yaml`'s `identical_output_check.compare_pre_threshold: true`: the
self-check compares `sigmoid(logits)` float32 arrays, not the binarised masks. Two different
models can legitimately binarise to the same mask on an easy image by coincidence; that is not
model collapse. Two models producing bit-identical *probabilities* on a real (non-degenerate)
input essentially never happens unless something is structurally wrong (shared weights, a
training bug, or a copy-paste checkpoint bug) — which is exactly the failure master §0b names.
`identical` = `np.array_equal(probs_a, probs_b)` (atol 0, matching evaluation.yaml).

---

## 1. `GET /api/health`

Liveness + model-load status. Called by the frontend before anything else; if this fails, the UI
shows a connection-error state (not a blank screen pretending to be empty data).

```jsonc
{
  "status": "ok",                       // "ok" | "degraded"
  "models_loaded": ["vanilla", "A", "B", "B_prime", "C"],
  "models_unavailable": [],             // e.g. ["C"] if a checkpoint failed to load
  "unavailable_reasons": {},            // {"C": "checkpoint file not found at ..."}
  "test_images_available": 98,
  "started_at_utc": "2026-08-09T12:00:00Z"
}
```

If any expected model failed to load, it is **excluded from every other endpoint's responses**,
never faked. `status` becomes `"degraded"` and `models_unavailable` names which. The frontend
must render this honestly (a visibly missing column), not silently hide the gap.

---

## 2. `GET /api/models`

Static per-model metadata, computed once at startup from the real loaded checkpoints (never
hardcoded — `params_count` from `model.count_parameters()`, `model_size_bytes` from
`Path(checkpoint_path).stat().st_size`, `checkpoint_hash` from `checkpointing.checkpoint_hash()`).

```jsonc
{
  "models": [
    {
      "model_id": "vanilla",            // one of: vanilla | A | B | B_prime | C
      "label": "Vanilla",
      "is_reference": false,            // true only for "A" -- relative bars are computed vs A
      "line_style": "dotted",           // solid | dotted | dashed | dashdot | longdash | solid_thick
                                         // -- matches src/visualization/contours.py::LINE_STYLES exactly
      "pruning_ratio": null,            // null for vanilla/A; 0.5 for B/B_prime/C
      "params_count": 31036481,
      "model_size_bytes": 124345872,
      "checkpoint_hash": "8f2a...c3",
      "checkpoint_path": "checkpoints/vanilla_seed1/best.pt",
      "seed": 1
    }
    // ... A, B, B_prime, C
  ]
}
```

`is_reference: true` on `"A"` is what tells the frontend which column's numbers are the
denominator for the relative bars (master §14: "Metrics as relative bars vs Model A").

---

## 3. `GET /api/images`

For the dropdown. Test-split only, labelled by curvature bin (master §14: "dropdown labeled by
curvature bin, default high-curvature").

```jsonc
{
  "default_image_id": "stardist_tcell/.../Training_source_....tif",  // a HIGH-bin image, chosen
                                                                       // deterministically (seed 42,
                                                                       // matching master §13's convention)
  "images": [
    {
      "image_id": "...",
      "dataset": "stardist_tcell",
      "curvature_bin": "high",          // low | mid | high
      "curvature_score": 0.2134
    }
    // ... every image in data/splits/test.csv, all three bins
  ]
}
```

The frontend groups this list into three `<optgroup>`s (Low / Mid / High curvature) and
pre-selects `default_image_id`.

---

## 4. `GET /api/case/{image_id}`

The primary endpoint: everything needed to render one full comparison screen, in one call —
original image, ground truth, and per-model prediction + metrics for every loaded model. Every
number in this response comes from a live forward pass computed for THIS request (or served from
a per-`(image_id, model_id)` cache populated by a real forward pass — never a fabricated value).

```jsonc
{
  "image_id": "...",
  "dataset": "stardist_tcell",
  "curvature_bin": "high",
  "curvature_score": 0.2134,
  "cell_size": 331.3,
  "contrast": 0.42,
  "original_png_base64": "iVBORw0KG...",     // 8-bit greyscale PNG, percentile-normalised (A20)
  "ground_truth": {
    "mask_png_base64": "iVBORw0KG...",       // 1-bit-equivalent (0/255) greyscale PNG
    "contours": [[[x, y], [x, y], ...], ...] // one array of [x,y] points per connected boundary
  },
  "models": [
    {
      "model_id": "vanilla",
      "prediction": {
        "mask_png_base64": "iVBORw0KG...",
        "contours": [[[x, y], ...], ...]
      },
      "metrics": {
        "dice": 0.8123,
        "boundary_f1_2px": 0.5511,
        "boundary_f1_5px": 0.7820,
        "hd95": 12.34                        // null (JSON) if undefined -- see below
      }
    }
    // ... A, B, B_prime, C -- only models NOT in models_unavailable
  ]
}
```

**Metrics are computed with the exact functions already implemented in
`src/metrics/segmentation.py`** (`dice_score`, `boundary_f1`, `hd95`) — the SAME code path used
for the study's own per-image tables (`src/metrics/per_image.py`), so a number shown in the UI
and a number in a results table can never silently disagree by using two different
implementations. `hd95` is JSON `null` (not `0` or `NaN`-as-string) when either mask is empty,
matching `segmentation.py`'s own documented NaN convention — the frontend renders this as "—",
never as a fabricated zero (a zero would misleadingly read as a perfect score).

**Contours** are vector polygons (pixel coordinates), not raster images, extracted server-side
(`cv2.findContours` + `cv2.approxPolyDP` to bound payload size is the expected Phase 11
implementation — a *new* function, since `src/visualization/contours.py`'s existing
`draw_contour`/`overlay_multiple_contours` are raster-oriented for the static paper figures,
Phase 13, a different consumer). Vector contours let the frontend render each model with a truly
distinct SVG `stroke-dasharray` — a real line-style difference, not a colour difference — and
toggle any one model's contour on/off without re-fetching.

**Error maps are NOT computed server-side.** The frontend derives FP/FN client-side from
`prediction.mask_png_base64` vs `ground_truth.mask_png_base64` (§6 below) — both real, already
fetched. This keeps the contract smaller and the error map trivially re-toggleable without a
network round trip, while remaining a live computation on real model output, not a fabrication.

---

## 5. `GET /api/selfcheck`

The master §14 startup self-check, cached from backend-startup inference (see §0).

```jsonc
{
  "probe_image_id": "...",              // fixed, deterministic -- same probe every startup
  "computed_at_utc": "2026-08-09T12:00:03Z",
  "collapsed": false,
  "pairs": [
    {"model_a": "vanilla", "model_b": "A", "identical": false, "mean_abs_diff": 0.1342},
    {"model_a": "vanilla", "model_b": "B", "identical": false, "mean_abs_diff": 0.0981}
    // ... all C(n,2) pairs among the models that actually loaded
  ]
}
```

`collapsed = any(pair.identical for pair in pairs)`. The frontend polls this once on page load
and renders the red **"MODEL COLLAPSE DETECTED"** banner, naming the colliding pair(s) by label,
whenever `collapsed` is `true`. This is not a soft warning the user can dismiss into irrelevance —
master's own account is that the previous attempt at this project failed exactly this way.

---

## 6. Client-side error-map algorithm (frontend responsibility, specified here for Phase 11 parity)

Given `pred` and `gt`, both decoded to 0/255 pixel arrays via `<canvas>.getImageData`:

```
for each pixel:
    pred=255, gt=0   -> false positive -> SOLID fill
    pred=0,   gt=255 -> false negative -> HATCHED fill
    otherwise        -> no fill (correct)
```

Rendered as two separate binary layers (`<canvas>` alpha masks), each independently toggleable,
composited over the greyscale image. Solid = filled black at reduced opacity; hatched = a
repeating diagonal-line pattern (CSS/canvas `repeating-linear-gradient`-equivalent), never colour
— master §14's "solid (FP) vs hatched (FN)" is a fill-pattern distinction, not a hue distinction.

---

## 7. Error responses

Any endpoint, on a bad `image_id` or `model_id`: HTTP 404,
`{"error": "image_id '...' not found in the test split"}`. On an internal failure during
inference: HTTP 500, `{"error": "..."}` — never a 200 with a fabricated/default body. The
frontend renders fetch failures as a visible error state per panel, never as silently blank or
zeroed metrics (which would be indistinguishable from a real score of 0).

---

## 8. What Phase 11 must NOT do

- Must not pre-compute and hardcode a static JSON fixture "for now" — every response is a live
  call into the loaded PyTorch models.
- Must not silently substitute one model's output for another's if a checkpoint fails to load —
  exclude it from `models`/`models_unavailable` instead (§1).
- Must not skip the startup self-check "to save time" — it is the one thing standing between this
  project and repeating its predecessor's exact failure mode.
