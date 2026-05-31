"""
FastAPI app — P2 owns this file.

Single endpoint: POST /api/predict
Accepts audio file upload, returns prediction JSON.

Usage: uvicorn api.main:app --reload --port 8000
"""
import os
import sys
import tempfile
import traceback

print("[SickNote] Starting API server...", flush=True)

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

print("[SickNote] FastAPI imported OK", flush=True)

try:
    from api.inference import predict
    print("[SickNote] Inference module imported OK", flush=True)
except Exception as e:
    print(f"[SickNote] FATAL: Failed to import inference: {e}", flush=True)
    traceback.print_exc()
    sys.exit(1)

app = FastAPI(title="SickNote API")

CORS_ORIGINS = os.environ.get(
    "CORS_ORIGINS",
    "http://localhost:3000,http://localhost:3001",
).split(",")

print(f"[SickNote] CORS origins: {CORS_ORIGINS}", flush=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".webm", ".wav", ".ogg", ".mp3"}


@app.get("/")
async def health():
    return {"status": "ok"}


@app.post("/api/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {ext}")

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = predict(tmp_path)
    finally:
        os.unlink(tmp_path)

    return result
