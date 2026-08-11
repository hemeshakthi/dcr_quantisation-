// dCR Compression Study -- live comparison UI orchestration.
//
// EVERY image and metric shown on screen comes from a parsed API response object (api.js /
// docs/api_contract.md). There is no local table of scores, no seeded/default metric numbers,
// and no code path that displays a number or picture without having fetched it first. Loading
// and error states render as "—" or a visible message, never as 0 or a placeholder value -- 0
// is a real, meaningful score and must never be confused with "no data yet".
//
// Prediction masks, contour overlays and error maps are SERVER-RENDERED PNGs (A39,
// docs/api_contract.md §0/§3): this file only places <img> elements and never draws a contour
// or computes a false-positive/false-negative pixel itself. An earlier draft of this file did
// that client-side (canvas pixel comparison + SVG polylines); POST /api/infer now returns
// ready-made overlay_png_base64 / error_map_png_base64 per model, rendered by
// src/visualization/contours.py so the live demo and the paper's static figures share one
// rendering implementation.

const HIGHER_IS_BETTER = {
  dice: true,
  boundary_f1_2px: true,
  boundary_f1_5px: true,
  hd95: false,
};
const METRIC_ORDER = ["dice", "boundary_f1_2px", "boundary_f1_5px", "hd95"];
const BAR_SCALE = 1.5; // ratio-to-reference at which a metric bar is visually full

// ------------------------------------------------------------------ small render helpers
function fmtMetric(v) {
  return v === null || v === undefined ? "—" : v.toFixed(4);
}
function fmtSigned(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  const s = v >= 0 ? "+" : "";
  return `${s}${v.toFixed(4)}`;
}
function fmtParamCount(n) {
  if (n === null || n === undefined) return "—";
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}k`;
  return String(n);
}
function fmtBytes(n) {
  if (n === null || n === undefined) return "—";
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)} MB`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)} kB`;
  return `${n} B`;
}
function labelFor(modelsById, modelId) {
  return modelsById[modelId] ? modelsById[modelId].label : modelId;
}
function png(base64) {
  return base64 ? `data:image/png;base64,${base64}` : "";
}

// ------------------------------------------------------------------ app state
const state = {
  modelsById: {},          // model_id -> entry from /api/health's `models`
  referenceModelId: null,  // the model_id with is_reference === true (Model A)
  cardEls: {},              // model_id -> { article, predImg, overlayImg, errorImg, metricEls }
  currentCurvatureBin: null,
};

// ------------------------------------------------------------------ DOM refs
const el = (id) => document.getElementById(id);
const collapseBanner = el("collapse-banner");
const collapseText = el("collapse-text");
const connectionError = el("connection-error");
const connectionDetail = el("connection-detail");
const modelsStatus = el("models-status");
const imageSelect = el("image-select");
const imageMeta = el("image-meta");
const originalImg = el("original-img");
const gtOverlayImg = el("gt-overlay-img");
const gtMaskImg = el("gt-mask-img");
const modelCardsSection = el("model-cards");
const cardTemplate = el("model-card-template");
const storySection = el("story");
const storyText = el("story-text");

function showConnectionError(message) {
  el("api-base-shown").textContent = API.base || window.location.origin;
  connectionDetail.textContent = message || "";
  connectionError.hidden = false;
}
function hideConnectionError() {
  connectionError.hidden = true;
}

// ------------------------------------------------------------------ collapse banner (Golden Rule 2)
function renderCollapseCheck(collapseCheck) {
  if (!collapseCheck || !collapseCheck.collapsed) {
    collapseBanner.hidden = true;
    return;
  }
  const bad = collapseCheck.pairs.filter((p) => p.identical);
  const names = bad
    .map((p) => `${labelFor(state.modelsById, p.model_a)} ≡ ${labelFor(state.modelsById, p.model_b)}`)
    .join(", ");
  collapseText.textContent =
    `MODEL COLLAPSE DETECTED — identical predictions on the probe image: ${names}. ` +
    `Probe image: ${collapseCheck.probe_image_id}. Results below cannot be trusted until this is fixed.`;
  collapseBanner.hidden = false;
}

// ------------------------------------------------------------------ honest degraded-mode notice
// The CURRENT real repo state is 0/5 checkpoints (Kaggle training has not yet produced any) --
// this must be reported plainly (Golden Rule 4), not hidden behind an empty grid that looks like
// a bug rather than an accurate "nothing trained yet" status.
function renderModelsStatus(health) {
  const missing = health.models_unavailable || [];
  if (!missing.length) {
    modelsStatus.hidden = true;
    return;
  }
  const reasons = missing.map((mid) => `${mid}: ${health.unavailable_reasons[mid] || "unavailable"}`);
  modelsStatus.textContent =
    `${missing.length}/5 model checkpoint(s) not loaded — ${reasons.join(" · ")}`;
  modelsStatus.hidden = false;
}

// ------------------------------------------------------------------ model cards (static shell)
function buildModelCards(models) {
  state.modelsById = {};
  modelCardsSection.innerHTML = "";
  state.cardEls = {};

  for (const m of models) {
    state.modelsById[m.model_id] = m;
    if (m.is_reference) state.referenceModelId = m.model_id;

    const node = cardTemplate.content.cloneNode(true);
    const article = node.querySelector(".model-card");
    if (m.is_reference) article.classList.add("is-reference");

    node.querySelector(".model-label").textContent = m.label;
    const ratioText = m.pruning_ratio === null || m.pruning_ratio === undefined
      ? "uncompressed"
      : `pruned ${Math.round(m.pruning_ratio * 100)}%`;
    node.querySelector(".model-static-meta").textContent =
      `${ratioText} · seed ${m.seed} · line style: ${m.line_style} · ` +
      `ckpt ${String(m.checkpoint_hash || "").slice(0, 10)}`;

    const predImg = node.querySelector(".pred-img");
    const overlayImg = node.querySelector(".overlay-img");
    const errorImg = node.querySelector(".error-img");

    const metricEls = {};
    for (const row of node.querySelectorAll(".metric-row")) {
      metricEls[row.dataset.metric] = {
        row,
        value: row.querySelector(".metric-value"),
        fill: row.querySelector(".metric-bar-fill"),
      };
    }
    // static fields, populated once here (real, from /api/health -- never per-image)
    metricEls.params_count.value.textContent = fmtParamCount(m.params_count);
    metricEls.model_size.value.textContent = fmtBytes(m.model_size_bytes);

    modelCardsSection.appendChild(node);
    state.cardEls[m.model_id] = { article, predImg, overlayImg, errorImg, metricEls };
  }
}

// ------------------------------------------------------------------ metrics + relative bars
function isWorseThanReference(metric, value, referenceValue) {
  if (value === null || referenceValue === null || value === undefined || referenceValue === undefined) {
    return false;
  }
  return HIGHER_IS_BETTER[metric] ? value < referenceValue : value > referenceValue;
}

function renderMetrics(modelId, metrics, referenceMetrics) {
  const { metricEls } = state.cardEls[modelId];
  for (const metric of METRIC_ORDER) {
    const value = metrics[metric];
    const refValue = referenceMetrics ? referenceMetrics[metric] : null;
    const { row, value: valueEl, fill: fillEl } = metricEls[metric];

    valueEl.textContent = fmtMetric(value);
    row.classList.toggle("worse-than-a", isWorseThanReference(metric, value, refValue));

    if (value === null || value === undefined || !refValue) {
      fillEl.style.width = "0%";
      continue;
    }
    const ratio = value / refValue;
    const pct = Math.min(ratio, BAR_SCALE) / BAR_SCALE * 100;
    fillEl.style.width = `${pct}%`;
    fillEl.parentElement.style.setProperty("--bar-scale", String(BAR_SCALE));
  }
  // curvature bin is a property of the IMAGE, echoed per card per docs/api_contract.md
  metricEls.curvature_bin.value.textContent = state.currentCurvatureBin || "—";
}

// ------------------------------------------------------------------ the honest 60-second story
// Every sentence here is CONDITIONAL on the real fetched numbers. There is no branch that
// asserts a positive result unconditionally -- if C does not beat B (or B'), that is exactly
// what gets printed (master §14: "If C doesn't beat B, show it honestly").
function computeStory(caseData) {
  const byId = Object.fromEntries(caseData.models.map((m) => [m.model_id, m]));
  const a = byId.A, b = byId.B, c = byId.C, bp = byId.B_prime;
  if (!a || !b || !c) {
    return { text: "Not enough models loaded (need at least A, B and C) to tell the recovery story for this image.", isNull: true };
  }
  const diceDrop = a.metrics.dice - b.metrics.dice;
  const f1Drop = a.metrics.boundary_f1_2px - b.metrics.boundary_f1_2px;
  const parts = [];
  parts.push(
    `Pruning (A→B) moved Dice by ${fmtSigned(-diceDrop)} but Boundary F1 (2px) by ${fmtSigned(-f1Drop)}` +
    (Math.abs(f1Drop) > Math.abs(diceDrop)
      ? " — on this image, Dice alone would have hidden most of the boundary damage."
      : " — on this image the two metrics moved by comparable amounts.")
  );

  const recovered = c.metrics.boundary_f1_2px - b.metrics.boundary_f1_2px;
  let isNull = false;
  if (recovered > 0) {
    parts.push(`dCR (B→C) recovered Boundary F1 (2px) by ${fmtSigned(recovered)} on this image.`);
  } else {
    parts.push(`dCR (B→C) did NOT improve Boundary F1 (2px) on this image (Δ=${fmtSigned(recovered)}).`);
    isNull = true;
  }

  if (bp) {
    const vsGeneric = c.metrics.boundary_f1_2px - bp.metrics.boundary_f1_2px;
    if (vsGeneric > 0) {
      parts.push(`C also beat the generic boundary-loss control B′ by ${fmtSigned(vsGeneric)}, consistent with a curvature-specific effect (P5).`);
    } else {
      parts.push(`C did NOT beat the generic control B′ here (Δ=${fmtSigned(vsGeneric)}) — any recovery on this image is not shown to be curvature-specific.`);
      isNull = true;
    }
  }

  parts.push("This is ONE test image, shown live — not the study's statistical result. See results/tables/ (Phase 8) for the confirmatory analysis across the full test set and all seeds.");
  return { text: parts.join(" "), isNull };
}

// ------------------------------------------------------------------ loading one case (image)
async function loadCase(imageId) {
  imageSelect.disabled = true;
  try {
    const caseData = await API.infer(imageId);
    state.currentCurvatureBin = caseData.curvature_bin;

    originalImg.src = png(caseData.original_png_base64);
    gtOverlayImg.src = png(caseData.ground_truth.overlay_png_base64);
    gtMaskImg.src = png(caseData.ground_truth.mask_png_base64);

    imageMeta.textContent =
      `${caseData.dataset} · curvature=${caseData.curvature_bin} (${caseData.curvature_score.toFixed(4)}) ` +
      `· cell size=${caseData.cell_size.toFixed(1)}px² · contrast=${caseData.contrast.toFixed(3)}` +
      (caseData.models.length ? "" : " · no models loaded, showing original/ground-truth only");

    const referenceEntry = caseData.models.find((m) => m.model_id === state.referenceModelId);
    const referenceMetrics = referenceEntry ? referenceEntry.metrics : null;

    for (const m of caseData.models) {
      const cardEl = state.cardEls[m.model_id];
      if (!cardEl) continue; // present in this response but not in /api/health's models -- shouldn't happen
      cardEl.predImg.src = png(m.prediction_mask_png_base64);
      cardEl.overlayImg.src = png(m.overlay_png_base64);
      cardEl.errorImg.src = png(m.error_map_png_base64);
      renderMetrics(m.model_id, m.metrics, referenceMetrics);
    }

    const story = computeStory(caseData);
    storyText.textContent = story.text;
    storyText.parentElement.classList.toggle("null-result", story.isNull);
    storySection.hidden = false;
  } catch (err) {
    imageMeta.textContent = `failed to load this image: ${err.message}`;
    throw err;
  } finally {
    imageSelect.disabled = false;
  }
}

// ------------------------------------------------------------------ image dropdown
function buildImageDropdown(imagesResponse) {
  imageSelect.innerHTML = "";
  const groups = { high: [], mid: [], low: [] };
  for (const img of imagesResponse.images) {
    (groups[img.curvature_bin] || (groups[img.curvature_bin] = [])).push(img);
  }
  const labels = { high: "High curvature", mid: "Mid curvature", low: "Low curvature" };
  for (const bin of ["high", "mid", "low"]) {
    if (!groups[bin] || !groups[bin].length) continue;
    const og = document.createElement("optgroup");
    og.label = `${labels[bin]} (${groups[bin].length})`;
    for (const img of groups[bin]) {
      const opt = document.createElement("option");
      opt.value = img.image_id;
      opt.textContent = `${img.image_id}  ·  score=${img.curvature_score.toFixed(4)}`;
      og.appendChild(opt);
    }
    imageSelect.appendChild(og);
  }
  imageSelect.value = imagesResponse.default_image_id;
  imageSelect.disabled = false;
  imageSelect.addEventListener("change", () => {
    loadCase(imageSelect.value).catch((e) => console.error("load case failed:", e));
  });
}

// ------------------------------------------------------------------ boot
async function main() {
  let health;
  try {
    health = await API.health();
  } catch (err) {
    showConnectionError(err.message);
    return; // nothing else can render honestly without a reachable backend
  }
  hideConnectionError();

  buildModelCards(health.models);
  renderCollapseCheck(health.collapse_check);
  renderModelsStatus(health);

  const imagesResponse = await API.images();
  buildImageDropdown(imagesResponse);
  if (imagesResponse.default_image_id) {
    await loadCase(imagesResponse.default_image_id);
  } else {
    imageMeta.textContent = "No test images available.";
  }
}

main().catch((err) => {
  console.error("app init failed:", err);
  showConnectionError(err.message);
});
