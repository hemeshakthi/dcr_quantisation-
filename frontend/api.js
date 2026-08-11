// Thin fetch wrapper over the backend, matching docs/api_contract.md exactly (3 endpoints:
// GET /api/health, GET /api/images, POST /api/infer).
//
// Deliberately has NO fallback/default metric values anywhere. Every function either resolves
// with real data parsed from a real HTTP response, or rejects -- app.js is responsible for
// rendering the rejected/error state visibly (never substituting a placeholder number, which
// would be indistinguishable from a genuine 0 or empty result).

const API = (() => {
  // Same-origin by default (the backend serves the frontend and the API from one FastAPI app,
  // per docs/api_contract.md). Overridable for local development against another origin.
  const BASE = window.DCR_API_BASE || "";

  async function request(path, options) {
    let res;
    try {
      res = await fetch(BASE + path, options);
    } catch (networkErr) {
      throw new ApiError(`network error calling ${path}: ${networkErr.message}`, null, path);
    }
    let body;
    const text = await res.text();
    try {
      body = text ? JSON.parse(text) : {};
    } catch {
      throw new ApiError(`non-JSON response from ${path}`, res.status, path);
    }
    if (!res.ok) {
      // docs/api_contract.md §4: error bodies are {"error": "..."}, not FastAPI's default
      // {"detail": "..."} -- backend/main.py installs a custom exception handler for this.
      throw new ApiError(body.error || `HTTP ${res.status} from ${path}`, res.status, path);
    }
    return body;
  }

  class ApiError extends Error {
    constructor(message, status, path) {
      super(message);
      this.name = "ApiError";
      this.status = status;
      this.path = path;
    }
  }

  return {
    ApiError,
    base: BASE,
    // GET /api/health -- loaded models + their static metadata + the collapse probe, in one call
    health: () => request("/api/health", { headers: { Accept: "application/json" } }),
    // GET /api/images -- test-split image list, labelled by curvature bin
    images: () => request("/api/images", { headers: { Accept: "application/json" } }),
    // POST /api/infer -- live inference for every currently-loaded model on one test image
    infer: (imageId) =>
      request("/api/infer", {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify({ image_id: imageId }),
      }),
  };
})();
