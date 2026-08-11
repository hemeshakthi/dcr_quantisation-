# Frontend — Live Boundary Comparison UI

Plain HTML/CSS/vanilla JS. No build step, no framework, no `npm install`. This is a deliberate
choice, not an oversight: Node/npm are available on the dev machine, but every other part of
this project is pure Python with no JS toolchain anywhere, `backend/` (the FastAPI app) serves
static files directly, and unbundled source is what makes `audit/verify_frontend.py`'s grep-based
checks ("no hardcoded metric values", "no color in CSS") meaningful — a minified bundle would
defeat them.

## How it's served

`backend/main.py` mounts this directory as static files
(`app.mount("/", StaticFiles(directory="frontend", html=True))`) so the page and the `/api/*`
endpoints are same-origin and `api.js`'s default `BASE = ""` just works:

```
uvicorn backend.main:app --reload
```

For local development against a backend on a different origin, set `window.DCR_API_BASE` before
the scripts load:

```html
<script>window.DCR_API_BASE = "http://localhost:8000";</script>
<script src="api.js"></script>
```

Opening `index.html` directly (no backend running) shows the connection-error state — correctly,
since there is nothing to fetch from.

## Files

| File | Role |
|---|---|
| `index.html` | structure; a `<template>` for the per-model card, cloned once per loaded model |
| `style.css` | grayscale-only design system (CSS custom properties, all shades of grey) |
| `api.js` | thin `fetch` wrapper matching `docs/api_contract.md`'s 3 endpoints exactly; no fallback data |
| `app.js` | orchestration: dropdown, placing server-rendered images, relative bars, collapse banner, degraded-mode notice, the honest "story" summary |

## Design decisions worth knowing about

- **Overlays, prediction masks and error maps are SERVER-RENDERED PNGs (A39).** An earlier draft
  of this frontend computed error maps client-side (canvas pixel comparison against two fetched
  mask images) and drew contours as client-side SVG `<polyline>` elements. The backend-authoring
  phase's instruction explicitly listed "error maps base64" as part of `POST /api/infer`'s
  response, which requires server-side rendering — so `app.js` now just assigns
  `overlay_png_base64` / `error_map_png_base64` / `prediction_mask_png_base64` to `<img src>`.
  This is a genuine simplification: no canvas pixel loop, no SVG polyline construction, no
  `DASH_ARRAY`/`STROKE_WIDTH` constants to keep in sync with the backend. The rendering itself
  lives in `src/visualization/contours.py` (`draw_contour`, `error_map`) — the SAME functions
  Phase 13's static paper figures use, so the live demo and the paper cannot visually disagree.
- **The collapse banner is not red.** An earlier task that requested this UI said "red ...
  banner", but master §14 freezes the entire UI as black-and-white, and that same task's own
  verification step required no color in CSS. Those conflict; the FROZEN spec wins. Urgency is
  instead conveyed by inversion (black background / white text), a pulsing border, bold oversized
  type, and a `⚠` glyph — see the comment at the top of `style.css`.
- **A "models not loaded" notice is shown honestly, not hidden.** The current real repo state is
  0/5 trained checkpoints (Kaggle training has not yet produced any) — `app.js::renderModelsStatus`
  reads `/api/health`'s `models_unavailable`/`unavailable_reasons` and states plainly which
  checkpoints are missing and why, rather than silently rendering an empty model-cards grid that
  would look like a bug (Golden Rule 4: null/negative state must be reported, not suppressed).
  Ground truth and the original image still render even with zero models loaded, since
  `POST /api/infer` returns those regardless of `models_loaded`.
- **Relative bars encode raw magnitude honestly**: bar length is the true `value / referenceValue`
  ratio (capped for display, never silently inverted), and a `worse-than-a` hatched fill pattern
  — computed per-metric via `HIGHER_IS_BETTER`, since HD95 is lower-is-better while Dice/Boundary
  F1 are higher-is-better — marks whichever direction is actually worse. The raw numbers are
  always shown alongside, never only the bar.
- **The "story" text is entirely conditional on fetched numbers.** There is no code path that
  asserts C recovered boundary geometry unconditionally; `computeStory()` in `app.js` branches on
  the real sign of each computed delta and prints the null-result phrasing when C does not beat B
  or does not beat B′, per master §14's "if C doesn't beat B, show it honestly". It also always
  ends by stating this is one live image, not the study's statistical result.

## What this frontend does NOT do

Compute any pixel data itself. Every mask, overlay, error map and metric value shown on screen
was computed by `backend/` from a real checkpoint's real forward pass; this directory only
fetches and displays it (`docs/api_contract.md` is the binding contract).
