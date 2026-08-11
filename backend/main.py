"""FastAPI app entry point (docs/api_contract.md).

Wires the three routers to one shared ModelRegistry (services/model_registry.py) and serves
frontend/ as static files, since frontend/api.js defaults to a same-origin API.

    uvicorn backend.main:app --reload
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from backend.routers import health, images, infer

ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT / "frontend"

app = FastAPI(title="dCR Compression Study -- Live Inference API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def _http_exception_handler(request, exc: HTTPException) -> JSONResponse:
    """`{"error": "..."}` bodies (docs/api_contract.md §4) instead of FastAPI's default
    `{"detail": "..."}` shape."""
    return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})


app.include_router(health.router, prefix="/api")
app.include_router(images.router, prefix="/api")
app.include_router(infer.router, prefix="/api")

if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
