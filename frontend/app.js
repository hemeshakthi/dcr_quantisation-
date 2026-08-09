// dCR Compression Study -- live comparison UI orchestration.
//
// EVERY metric shown on screen is read from a parsed API response object (see api.js /
// docs/api_contract.md). There is no local table of scores, no seeded/default metric numbers,
// and no code path that displays a number without having fetched it first. Loading and error
// states render as "—" or a visible error message, never as 0 or a placeholder value -- 0 is a
// real, meaningful score and must never be confused with "no data yet".
//
// IMAGE_SIZE is a structural pipeline constant (Phase 2 preprocessing target, A19 --
// every processed image is 1024x1024), not a metric: it only tells the SVG viewBox what
// coordinate space the backend's contour points are already in.
const IMAGE_SIZE = 1024;

// Presentation-only constants (line dash patterns, hatch spacing). None of these are data.
// Names match src/visualization/contours.py::LINE_STYLES exactly, so a model's line style is
// never chosen independently on the two sides of the stack.
const DASH_ARRAY = {
  solid: "",
  solid_thick: "",
  dotted: "2,4",
  dashed: "9,5",
  dashdot: "9,4,2,4",
  longdash: "16,6",
};
const STROKE_WIDTH = { solid_thick: 3 };

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
  modelsById: {},          // model_id -> entry from /api/models
  referenceModelId: null,  // the model_id with is_reference === true (Model A)
  cardEls: {},             // model_id -> { article, predImg, canvas, svg, metricEls, toggles }
  errorMasks: {},          // model_id -> { fp: Uint8Array, fn: Uint8Array, w, h } for current case
};

// ------------------------------------------------------------------ DOM refs
const el = (id) => document.getElementById(id);
const collapseBanner = el("collapse-banner");
const collapseText = el("collapse-text");
const connectionError = el("connection-error");
const connectionDetail = el("connection-detail");
const imageSelect = el("image-select");
const imageMeta = el("image-meta");
const originalImg = el("original-img");
const gtImg = el("gt-img");
const gtContourSvg = el("gt-contour-svg");
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

// ------------------------------------------------------------------ self-check banner
function renderSelfCheck(selfcheck) {
  if (!selfcheck || !selfcheck.collapsed) {
    collapseBanner.hidden = true;
    return;
  }
  const bad = selfcheck.pairs.filter((p) => p.identical);
  const names = bad
    .map((p) => `${labelFor(state.modelsById, p.model_a)} ≡ ${labelFor(state.modelsById, p.model_b)}`)
    .join(", ");
  collapseText.textContent =
    `MODEL COLLAPSE DETECTED — identical predictions on the startup probe: ${names}. ` +
    `Probe image: ${selfcheck.probe_image_id}. Results below cannot be trusted until this is fixed.`;
  collapseBanner.hidden = false;
}

// ------------------------------------------------------------------ model cards (static shell)
function buildModelCards(modelsResponse) {
  state.modelsById = {};
  modelCardsSection.innerHTML = "";
  state.cardEls = {};

  for (const m of modelsResponse.models) {
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
      `${ratioText} · seed ${m.seed} · ckpt ${String(m.checkpoint_hash || "").slice(0, 10)}`;

    const svg = node.querySelector(".contour-svg");
    const predImg = node.querySelector(".pred-img");
    const canvas = node.querySelector(".error-canvas");

    const metricEls = {};
    for (const row of node.querySelectorAll(".metric-row")) {
      metricEls[row.dataset.metric] = {
        row,
        value: row.querySelector(".metric-value"),
        fill: row.querySelector(".metric-bar-fill"),
      };
    }
    // static fields, populated once here (real, from /api/models -- never per-image)
    metricEls.params_count.value.textContent = fmtParamCount(m.params_count);
    metricEls.model_size.value.textContent = fmtBytes(m.model_size_bytes);

    const toggles = {
      fp: node.querySelector(".toggle-fp"),
      fn: node.querySelector(".toggle-fn"),
      contour: node.querySelector(".toggle-contour"),
    };
    const redraw = () => drawErrorCanvas(m.model_id);
    toggles.fp.addEventListener("change", redraw);
    toggles.fn.addEventListener("change", redraw);
    toggles.contour.addEventListener("change", () => {
      svg.style.display = toggles.contour.checked ? "" : "none";
    });

    modelCardsSection.appendChild(node);
    state.cardEls[m.model_id] = { article, predImg, canvas, svg, metricEls, toggles };
  }
}

// ------------------------------------------------------------------ contours (vector, SVG)
function renderContours(svgEl, contours, lineStyle) {
  svgEl.innerHTML = "";
  if (!contours) return;
  const dash = DASH_ARRAY[lineStyle] || "";
  const width = STROKE_WIDTH[lineStyle] || 1.6;
  for (const poly of contours) {
    const pl = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
    pl.setAttribute("points", poly.map(([x, y]) => `${x},${y}`).join(" "));
    if (dash) pl.setAttribute("stroke-dasharray", dash);
    pl.setAttribute("stroke-width", String(width));
    svgEl.appendChild(pl);
  }
}

// ------------------------------------------------------------------ error map (client-side,
// per docs/api_contract.md §6): computed ONCE per model when a case loads, from the two REAL
// fetched masks; toggling FP/FN/contour only changes what's drawn from the precomputed masks,
// never re-derives them from anything but the fetched pixel data.
async function decodeMaskToBinary(base64Png) {
  const img = new Image();
  img.src = png(base64Png);
  await img.decode();
  const c = document.createElement("canvas");
  c.width = img.naturalWidth;
  c.height = img.naturalHeight;
  const ctx = c.getContext("2d");
  ctx.drawImage(img, 0, 0);
  const { data, width, height } = ctx.getImageData(0, 0, c.width, c.height);
  const bin = new Uint8Array(width * height);
  for (let i = 0; i < bin.length; i++) bin[i] = data[i * 4] >= 128 ? 1 : 0; // R channel; source is greyscale
  return { bin, width, height };
}

async function computeErrorMasks(predBase64, gtBase64) {
  const [pred, gt] = await Promise.all([decodeMaskToBinary(predBase64), decodeMaskToBinary(gtBase64)]);
  const n = pred.bin.length;
  const fp = new Uint8Array(n);
  const fn = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    if (pred.bin[i] === 1 && gt.bin[i] === 0) fp[i] = 1;
    else if (pred.bin[i] === 0 && gt.bin[i] === 1) fn[i] = 1;
  }
  return { fp, fn, w: pred.width, h: pred.height };
}

function drawErrorCanvas(modelId) {
  const cardEl = state.cardEls[modelId];
  const masks = state.errorMasks[modelId];
  if (!cardEl || !masks) return;
  const { canvas, toggles } = cardEl;
  canvas.width = masks.w;
  canvas.height = masks.h;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, masks.w, masks.h);
  const imgData = ctx.createImageData(masks.w, masks.h);
  const showFP = toggles.fp.checked;
  const showFN = toggles.fn.checked;
  for (let y = 0; y < masks.h; y++) {
    for (let x = 0; x < masks.w; x++) {
      const i = y * masks.w + x;
      let on = false, alpha = 0;
      if (showFP && masks.fp[i]) {
        on = true; alpha = 140; // FP: dense/solid fill
      } else if (showFN && masks.fn[i]) {
        // FN: HATCHED -- only a diagonal-stripe subset of pixels is painted, giving a real
        // fill-pattern distinction from FP rather than a colour distinction (master §14).
        if (((x + y) % 8) < 3) { on = true; alpha = 180; }
      }
      if (on) {
        const o = i * 4;
        imgData.data[o] = 0; imgData.data[o + 1] = 0; imgData.data[o + 2] = 0;
        imgData.data[o + 3] = alpha;
      }
    }
  }
  ctx.putImageData(imgData, 0, 0);
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
    const caseData = await API.case(imageId);
    state.currentCurvatureBin = caseData.curvature_bin;

    originalImg.src = png(caseData.original_png_base64);
    gtImg.src = png(caseData.ground_truth.mask_png_base64);
    renderContours(gtContourSvg, caseData.ground_truth.contours, "solid");

    imageMeta.textContent =
      `${caseData.dataset} · curvature=${caseData.curvature_bin} (${caseData.curvature_score.toFixed(4)}) ` +
      `· cell size=${caseData.cell_size.toFixed(1)}px² · contrast=${caseData.contrast.toFixed(3)}`;

    const referenceEntry = caseData.models.find((m) => m.model_id === state.referenceModelId);
    const referenceMetrics = referenceEntry ? referenceEntry.metrics : null;

    for (const m of caseData.models) {
      const cardEl = state.cardEls[m.model_id];
      if (!cardEl) continue; // model present in this case but not in /api/models -- shouldn't happen
      cardEl.predImg.src = png(m.prediction.mask_png_base64);
      const lineStyle = state.modelsById[m.model_id]
        ? state.modelsById[m.model_id].line_style
        : "solid";
      renderContours(cardEl.svg, m.prediction.contours, lineStyle);
      renderMetrics(m.model_id, m.metrics, referenceMetrics);

      // eslint-disable-next-line no-await-in-loop -- decoding is cheap and order doesn't matter here
      state.errorMasks[m.model_id] = await computeErrorMasks(
        m.prediction.mask_png_base64, caseData.ground_truth.mask_png_base64);
      drawErrorCanvas(m.model_id);
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
  try {
    await API.health();
  } catch (err) {
    showConnectionError(err.message);
    return; // nothing else can render honestly without a reachable backend
  }
  hideConnectionError();

  const [modelsResponse, imagesResponse, selfcheck] = await Promise.all([
    API.models(),
    API.images(),
    API.selfcheck().catch((err) => {
      // A failed self-check call is itself worth surfacing loudly -- it is the one check
      // standing between this project and repeating its predecessor's collapse failure.
      console.error("selfcheck failed:", err);
      return null;
    }),
  ]);

  buildModelCards(modelsResponse);
  renderSelfCheck(selfcheck);
  buildImageDropdown(imagesResponse);
  await loadCase(imagesResponse.default_image_id);
}

main().catch((err) => {
  console.error("app init failed:", err);
  showConnectionError(err.message);
});
