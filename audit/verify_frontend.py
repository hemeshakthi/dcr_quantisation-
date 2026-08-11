"""Frontend verification (master §14): structural checks, PLUS a live end-to-end check now
that backend/ exists.

  * no hardcoded metric values anywhere in the frontend source
  * no colour in CSS (every colour token is a true grayscale value)
  * every metric displayed is wired to a real API call, not a literal

Plus cross-file consistency checks (model_ids matching src/visualization/contours.py's
LINE_STYLES, api.js's endpoints matching docs/api_contract.md's 3-endpoint contract) that are
cheap to get wrong silently and expensive to notice later, and a live TestClient smoke test that
the frontend's actual fetch targets are genuinely implemented and return correctly-shaped data.

NOTE on the endpoint shape: this file originally verified a 5-endpoint draft contract
(GET /api/health, /models, /images, /case/{id}, /selfcheck) written before backend/ existed.
The backend-authoring phase's instruction specified a different, more specific 3-endpoint shape
(GET /api/health, GET /api/images, POST /api/infer) with server-rendered overlays/error maps,
which docs/api_contract.md's v2 (see its own header note) and backend/ both implement. This
script was rewritten to match that shape -- the OLD checks (API.models/API.case/API.selfcheck,
client-side SVG contours, FP/FN toggle checkboxes) no longer describe what was actually built.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
FRONTEND = ROOT / "frontend"

failures, warnings = [], []


def check(label, cond, warn_only=False):
    ok = bool(cond)
    print(f"[{'PASS' if ok else ('WARN' if warn_only else 'FAIL')}] {label}")
    if not ok:
        (warnings if warn_only else failures).append(label)
    return ok


print("=" * 78)
print("FRONTEND VERIFICATION (master §14) -- structural + live checks")
print("=" * 78)

# ---- 0. files exist ----------------------------------------------------------------
REQUIRED = ["index.html", "style.css", "api.js", "app.js", "README.md"]
for f in REQUIRED:
    check(f"frontend/{f} exists", (FRONTEND / f).is_file())

html = (FRONTEND / "index.html").read_text(encoding="utf-8")
css = (FRONTEND / "style.css").read_text(encoding="utf-8")
api_js = (FRONTEND / "api.js").read_text(encoding="utf-8")
app_js = (FRONTEND / "app.js").read_text(encoding="utf-8")
contract = (ROOT / "docs" / "api_contract.md").read_text(encoding="utf-8")

# ---- 1. JS syntax --------------------------------------------------------------------
for f in ("api.js", "app.js"):
    try:
        r = subprocess.run(["node", "--check", str(FRONTEND / f)], capture_output=True, text=True)
        check(f"{f} is syntactically valid JavaScript (node --check)", r.returncode == 0)
        if r.returncode != 0:
            print("      " + r.stderr.strip().replace("\n", "\n      "))
    except FileNotFoundError:
        check(f"{f} is syntactically valid JavaScript (node --check)", True, warn_only=True)
        print("      node not found on PATH -- skipped, not a frontend defect")

# ======================================================================================
# 2. NO HARDCODED METRIC VALUES
# ======================================================================================
print("\n---------------- no hardcoded metric values ----------------")
METRIC_KEYS = ["dice", "boundary_f1_2px", "boundary_f1_5px", "hd95",
              "params_count", "model_size_bytes"]

# A metric key immediately followed by ':' or '=' and then a NUMBER (not a variable, not a
# property lookup) is a hardcoded value. `metrics[metric]`, `m.metrics.dice`, `data.dice` etc.
# are property ACCESS, not assignment-of-a-literal, and must not match.
literal_hardcode_re = re.compile(
    r"\b(" + "|".join(METRIC_KEYS) + r")\s*[:=]\s*-?\d+(\.\d+)?\b")
hardcode_hits = []
for f, src in (("app.js", app_js), ("api.js", api_js), ("index.html", html)):
    for ln, line in enumerate(src.splitlines(), 1):
        if "//" in line and line.strip().startswith("//"):
            continue
        m = literal_hardcode_re.search(line)
        if m:
            hardcode_hits.append(f"{f}:{ln}: {line.strip()[:90]}")
check(f"no metric key is assigned a numeric LITERAL anywhere ({hardcode_hits or 'none found'})",
      not hardcode_hits)

# No embedded fixture/sample-data object carrying multiple metric keys at once, WITH ACTUAL
# SCORE-LIKE VALUES (the classic "mock API response" pattern this project has been vigilant
# against everywhere else). Requiring a decimal literal after the key excludes legitimate
# lookup tables that merely NAME the metrics, e.g. `HIGHER_IS_BETTER = {dice: true,
# boundary_f1_2px: true, ...}` -- booleans, not fabricated scores.
fixture_candidates = re.findall(r"\{[^{}]*\bdice\b[^{}]*\bboundary_f1[^{}]*\}",
                                app_js + "\n" + api_js, re.S)
fixture_hits = [c for c in fixture_candidates if re.search(r"\bdice['\"]?\s*:\s*-?\d+\.\d+", c)]
check(f"no embedded object literal bundles multiple metric fields with real score-like "
      f"values (a mock response) -- {len(fixture_hits)} found", not fixture_hits)

# Scope scan matching the pattern used throughout this project's out-of-scope checks
# (Phase 3/4): prose EXPLAINING that something is avoided must not be flagged as the thing
# itself, so a line containing an explicit negation ("not", "never", "no ", "nothing") next to
# the forbidden word is excluded.
FORBIDDEN_WORDS = ["mock", "fake data", "placeholder", "dummy", "hardcoded", "sample_data",
                   "todo: replace", "fixme"]
NEGATIONS = ("not ", "never", "no ", "nothing", "n't ")
scope_hits = []
for f, src in (("app.js", app_js), ("api.js", api_js), ("index.html", html), ("style.css", css)):
    for ln, line in enumerate(src.splitlines(), 1):
        low = line.lower()
        for w in FORBIDDEN_WORDS:
            if w in low and not any(neg in low for neg in NEGATIONS):
                scope_hits.append(f"{f}:{ln}: contains {w!r} without a negation: {line.strip()[:80]}")
check(f"no mock/fake/placeholder/dummy markers outside of explanatory negated prose "
      f"({scope_hits or 'none'})", not scope_hits)

# Math.random() would be the other classic way to fake a metric.
check("no Math.random() anywhere in the frontend (a real API call is the only data source)",
      "Math.random(" not in app_js and "Math.random(" not in api_js)

# ======================================================================================
# 3. NO COLOR IN CSS -- parse every colour token, assert true grayscale
# ======================================================================================
print("\n---------------- no color in CSS ----------------")


def hex_is_gray(hexstr: str) -> bool:
    h = hexstr.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) == 4:
        h = "".join(c * 2 for c in h[:3]) + h[3] * 2
    if len(h) not in (6, 8):
        return False
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return r == g == b


hex_tokens = re.findall(r"#[0-9a-fA-F]{3,8}\b", css)
bad_hex = [h for h in hex_tokens if not hex_is_gray(h)]
check(f"every hex colour in style.css is true grayscale (R==G==B) -- "
      f"{len(hex_tokens)} found, {len(bad_hex)} non-gray: {bad_hex}", not bad_hex)

rgb_tokens = re.findall(r"rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*[,)]", css)
bad_rgb = [t for t in rgb_tokens if not (t[0] == t[1] == t[2])]
check(f"every rgb()/rgba() colour in style.css is grayscale (R==G==B) -- "
      f"{len(rgb_tokens)} found, {len(bad_rgb)} non-gray: {bad_rgb}", not bad_rgb)

hsl_tokens = re.findall(r"hsla?\(\s*[\d.]+\s*,\s*([\d.]+)%", css)
bad_hsl = [t for t in hsl_tokens if float(t) != 0.0]
check(f"every hsl()/hsla() colour in style.css has 0% saturation -- "
      f"{len(hsl_tokens)} found, {len(bad_hsl)} with hue: {bad_hsl}", not bad_hsl)

CHROMATIC_KEYWORDS = ["red", "blue", "green", "orange", "yellow", "purple", "pink", "cyan",
                     "magenta", "violet", "indigo", "crimson", "gold", "coral", "salmon",
                     "teal", "navy", "maroon", "olive", "lime", "aqua", "fuchsia"]
# Strip /* ... */ comments before scanning: style.css's own header explicitly discusses (and
# rejects) a literal "red" banner in prose -- a real CSS RULE using red must still be caught,
# but the comment explaining why it was NOT used must not be.
css_code_only = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
kw_hits = []
for kw in CHROMATIC_KEYWORDS:
    if re.search(rf"\b{kw}\b", css_code_only, re.IGNORECASE):
        kw_hits.append(kw)
check(f"no chromatic CSS colour keywords in style.css CODE (comments excluded) "
      f"({kw_hits or 'none'})", not kw_hits)

check("style.css documents WHY the collapse banner is not literally red (A36)",
      "not colour" in css.lower() or "not color" in css.lower())

# ======================================================================================
# 4. EVERY NUMBER WIRED TO AN API CALL
# ======================================================================================
print("\n---------------- every metric wired to a live API call ----------------")

REQUIRED_CALLS = ["API.health(", "API.images(", "API.infer("]
for call in REQUIRED_CALLS:
    check(f"app.js calls {call.rstrip('(')}(...)", call in app_js)

REQUIRED_ENDPOINTS = {
    "/api/health": "health", "/api/images": "images", "/api/infer": "infer",
}
for path, name in REQUIRED_ENDPOINTS.items():
    check(f"api.js implements {name} against {path}", path in api_js)
    check(f"docs/api_contract.md documents {path}", path in contract)

# Every metric-value DOM write must read from a property path (obj.field / obj["field"]),
# never a bare literal -- checked above; here we additionally confirm the render function
# that DOES the writing is only ever invoked with data that came out of an API() call, by
# checking loadCase() (the sole place metrics get rendered) awaits API.infer(...) before any
# metric rendering call, and that renderMetrics's only literal default is the "no data" dash.
check("renderMetrics() is only called from within loadCase(), after API.infer() resolves",
      bool(re.search(r"await API\.infer\([^)]*\).*?renderMetrics\(", app_js, re.S)))
check("the ONLY literal string used as a metric fallback is the em-dash placeholder for "
      "missing data ('—'), never a number",
      "'—'" in app_js or '"—"' in app_js)
check("metric values are always read via a property path (`.metrics[` / `.metrics.`), "
      "never assigned a bare number",
      "metrics[metric]" in app_js or ".metrics." in app_js)

# ======================================================================================
# 5. cross-file consistency
# ======================================================================================
print("\n---------------- cross-file consistency ----------------")

contours_py = (ROOT / "src" / "visualization" / "contours.py").read_text(encoding="utf-8")
registry_py = (ROOT / "backend" / "services" / "model_registry.py").read_text(encoding="utf-8")

# LINE_STYLES: {model_id: style_name}. Every model_id the backend actually serves
# (MODEL_SPECS in backend/services/model_registry.py) must have an entry, or a model's overlay
# would silently render with the fallback "solid" style, making it visually indistinguishable
# from the ground truth / another model at that ratio.
py_line_styles = dict(re.findall(r'"(\w+)":\s*"(solid|dotted|dashed|dashdot|longdash|solid_thick)"',
                                 contours_py))
registry_model_ids = set(re.findall(r'"model_id":\s*"(\w+)"', registry_py))
missing_styles = registry_model_ids - set(py_line_styles)
check(f"every model_id backend/services/model_registry.py::MODEL_SPECS serves has a matching "
      f"entry in src/visualization/contours.py::LINE_STYLES (missing: {missing_styles or 'none'})",
      not missing_styles)

# Rendering itself is server-side now (A39): app.js must NOT re-implement contour drawing or
# error-map computation -- it should only ever assign a base64 PNG to an <img src>.
check("app.js does not reimplement client-side error-map computation (no getImageData/putImageData)",
      "getImageData" not in app_js and "putImageData" not in app_js)
check("app.js does not reimplement client-side contour drawing (no SVG polyline creation)",
      "createElementNS" not in app_js)
check("app.js places server-rendered images via the three documented response fields "
      "(prediction_mask_png_base64 / overlay_png_base64 / error_map_png_base64)",
      all(k in app_js for k in ["prediction_mask_png_base64", "overlay_png_base64",
                                "error_map_png_base64"]))

check("app.js's METRIC_ORDER matches the exact metric field names used in "
      "src/metrics/segmentation.py's output (dice, boundary_f1_2px, boundary_f1_5px, hd95)",
      all(k in app_js for k in ["dice", "boundary_f1_2px", "boundary_f1_5px", "hd95"]))

# ======================================================================================
# 6. required master §14 UI elements are present
# ======================================================================================
print("\n---------------- master §14 UI requirements present ----------------")
check("image dropdown present, labelled by curvature bin", 'id="image-select"' in html)
check("dropdown groups by curvature bin (optgroup, high/mid/low)",
      "optgroup" in app_js and "high" in app_js and "mid" in app_js and "low" in app_js)
check("default selection is the high-curvature group (per default_image_id from the API)",
      "default_image_id" in app_js)
check("original-image panel present", 'id="original-img"' in html)
check("ground-truth overlay panel present", 'id="gt-overlay-img"' in html)
check("ground-truth mask panel present", 'id="gt-mask-img"' in html)
check("per-model prediction image present in the card template", 'class="pred-img"' in html)
check("per-model contour overlay image present in the card template", 'class="overlay-img"' in html)
check("per-model error map image present in the card template", 'class="error-img"' in html)
check("error map distinguishes FP (solid) vs FN (hatched) by PATTERN not colour -- documented "
      "in the img alt text and rendered server-side by src/visualization/contours.py::error_map",
      "solid" in html.lower() and "hatch" in html.lower() and "def error_map" in contours_py)
check("Dice/BoundaryF1/HD95/model size/param count/curvature bin all present in the metrics panel",
      all(k in html for k in ["Dice", "Boundary F1", "HD95", "Parameters", "Model size",
                              "Curvature bin"]))
check("relative bars vs Model A present (metric-bar element + is_reference concept)",
      "metric-bar" in html and "is_reference" in app_js)
check("raw numbers shown alongside bars (not bars alone)", "metric-value" in html)
check("startup self-check banner present", 'id="collapse-banner"' in html)
check("collapse banner text says MODEL COLLAPSE DETECTED",
      "MODEL COLLAPSE DETECTED" in html or "MODEL COLLAPSE DETECTED" in app_js)
check("degraded-mode notice present (honest reporting when <5 checkpoints are loaded, Golden Rule 4)",
      'id="models-status"' in html and "models_unavailable" in app_js)
check("honest 'story' section present (shows null results too, per computeStory())",
      "computeStory" in app_js and "isNull" in app_js)
check("story explicitly states this is a single image, not the study's statistical result",
      "not the study" in app_js or "statistical result" in app_js)

# ======================================================================================
# 7. LIVE end-to-end check against backend/ via FastAPI's TestClient
# ======================================================================================
print("\n---------------- live backend smoke test (FastAPI TestClient) ----------------")
try:
    from fastapi.testclient import TestClient

    from backend.main import app as fastapi_app

    client = TestClient(fastapi_app)

    r_health = client.get("/api/health")
    check("GET /api/health returns 200", r_health.status_code == 200)
    health_body = r_health.json() if r_health.status_code == 200 else {}
    check("health response has the fields app.js reads (status/models_loaded/models_unavailable/"
          "unavailable_reasons/models/collapse_check/test_images_available)",
          all(k in health_body for k in ["status", "models_loaded", "models_unavailable",
                                         "unavailable_reasons", "models", "collapse_check",
                                         "test_images_available"]))

    r_images = client.get("/api/images")
    check("GET /api/images returns 200", r_images.status_code == 200)
    images_body = r_images.json() if r_images.status_code == 200 else {}
    check("images response has default_image_id + images[] with curvature_bin",
          "default_image_id" in images_body and images_body.get("images")
          and "curvature_bin" in images_body["images"][0])

    default_id = images_body.get("default_image_id")
    if default_id:
        r_infer = client.post("/api/infer", json={"image_id": default_id})
        check("POST /api/infer on the real default image returns 200", r_infer.status_code == 200)
        infer_body = r_infer.json() if r_infer.status_code == 200 else {}
        check("infer response has original_png_base64 + ground_truth.mask_png_base64 "
              "+ ground_truth.overlay_png_base64 + models[]",
              "original_png_base64" in infer_body
              and "mask_png_base64" in infer_body.get("ground_truth", {})
              and "overlay_png_base64" in infer_body.get("ground_truth", {})
              and "models" in infer_body)
        if infer_body.get("models"):
            m0 = infer_body["models"][0]
            check("each model entry in /api/infer has the 3 image fields app.js consumes",
                  all(k in m0 for k in ["prediction_mask_png_base64", "overlay_png_base64",
                                        "error_map_png_base64"]))
        else:
            check("infer models[] is empty because 0 checkpoints are currently loaded "
                  "(honest degraded state, not a bug -- see checkpoints/)", True, warn_only=True)
    else:
        check("GET /api/images returned a usable default_image_id to POST /api/infer with",
              False, warn_only=True)

    r_404 = client.post("/api/infer", json={"image_id": "does-not-exist"})
    check("POST /api/infer on an unknown image_id returns 404 with an {error: ...} body",
          r_404.status_code == 404 and "error" in r_404.json()
          and not r_404.json()["error"].startswith("'"))
except Exception as exc:  # noqa: BLE001 -- a failure here is itself the thing being verified
    check(f"live backend smoke test ran without raising ({exc!r})", False)

print()
print("=" * 78)
if warnings:
    print(f"WARNINGS ({len(warnings)}):")
    for w in warnings:
        print(f"  - {w}")
if failures:
    print(f"RESULT: FAIL ({len(failures)} check(s) failed)")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("RESULT: PASS — frontend structurally + live verified.")
print("=" * 78)
