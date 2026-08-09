# Frontend — Live Boundary Comparison UI

Plain HTML/CSS/vanilla JS. No build step, no framework, no `npm install`. This is a deliberate
choice, not an oversight: Node/npm are available on the dev machine, but every other part of
this project is pure Python with no JS toolchain anywhere, the Phase 11 backend serves static
files directly from FastAPI, and unbundled source is what makes `audit/verify_frontend.py`'s
grep-based checks ("no hardcoded metric values", "no color in CSS") meaningful — a minified
bundle would defeat them.

## How it's served

Phase 11's FastAPI backend mounts this directory as static files (e.g.
`app.mount("/", StaticFiles(directory="frontend", html=True))`) so the page and the `/api/*`
endpoints are same-origin and `api.js`'s default `BASE = ""` just works. For local development
against a backend on a different origin, set `window.DCR_API_BASE` before the scripts load:

```html
<script>window.DCR_API_BASE = "http://localhost:8000";</script>
<script src="api.js"></script>
```

Until Phase 11 exists, opening `index.html` directly will show the connection-error state —
correctly, since there is no backend yet. That is expected: this phase defines the frontend and
the exact contract (`docs/api_contract.md`) it calls; wiring it to real inference is Phase 11.

## Files

| File | Role |
|---|---|
| `index.html` | structure; a `<template>` for the per-model card, cloned once per loaded model |
| `style.css` | grayscale-only design system (CSS custom properties, all shades of grey) |
| `api.js` | thin `fetch` wrapper matching `docs/api_contract.md` exactly; no fallback data |
| `app.js` | orchestration: dropdown, contours (SVG), error maps (canvas), relative bars, self-check banner, the honest "story" summary |

## Design decisions worth knowing about

- **The collapse banner is not red.** The task that requested this UI said "red ... banner",
  but master §14 freezes the entire UI as black-and-white, and this same task's own verification
  step requires no color in CSS. Those conflict; the FROZEN spec wins. Urgency is instead
  conveyed by inversion (black background / white text), a pulsing border, bold oversized type,
  and a `⚠` glyph — see the comment at the top of `style.css`.
- **Error maps are computed client-side**, from the two mask PNGs the case endpoint already
  returns, per `docs/api_contract.md` §6 — not a separate backend endpoint. FP is a dense solid
  fill; FN is a genuine pixel-level hatch pattern (only pixels where `(x+y) % 8 < 3` are painted),
  computed once per model per image load and redrawn (not recomputed) on toggle.
- **Contours are vector, not raster** — SVG `<polyline>` elements with `stroke-dasharray`
  matching `src/visualization/contours.py::LINE_STYLES` by name, so line style can never drift
  between the two.
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

## What this phase does NOT do

Implement the backend. `docs/api_contract.md` is the binding spec Phase 11 must satisfy; nothing
here fabricates a response to stand in for it.
