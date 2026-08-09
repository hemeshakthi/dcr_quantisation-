// Thin fetch wrapper over the Phase 11 backend, matching docs/api_contract.md exactly.
//
// Deliberately has NO fallback/default metric values anywhere. Every function either resolves
// with real data parsed from a real HTTP response, or rejects -- app.js is responsible for
// rendering the rejected/error state visibly (never substituting a placeholder number, which
// would be indistinguishable from a genuine 0 or empty result).

const API = (() => {
  // Same-origin by default (Phase 11 serves the frontend and API from one FastAPI app, per
  // docs/api_contract.md). Overridable for local development against a backend on another port.
  const BASE = window.DCR_API_BASE || "";

  async function getJSON(path) {
    let res;
    try {
      res = await fetch(BASE + path, { headers: { Accept: "application/json" } });
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
    // GET /api/health
    health: () => getJSON("/api/health"),
    // GET /api/models
    models: () => getJSON("/api/models"),
    // GET /api/images
    images: () => getJSON("/api/images"),
    // GET /api/case/{image_id}
    // image_id is a path segment but may itself contain '/', matching the dataset-prefixed
    // image_ids used throughout this project (e.g. "stardist_tcell/.../foo.tif") -- encode
    // each '/'-delimited component separately rather than escaping the whole id, so the
    // backend can route it as a single path parameter that still reads like the real id.
    case: (imageId) =>
      getJSON(`/api/case/${imageId.split("/").map(encodeURIComponent).join("/")}`),
    // GET /api/selfcheck
    selfcheck: () => getJSON("/api/selfcheck"),
  };
})();
