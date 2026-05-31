FROM python:3.12-slim

WORKDIR /app

# Install system deps for torchaudio
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only torch to keep image small (~400MB vs ~2GB with CUDA)
RUN pip install --no-cache-dir \
    torch==2.5.1+cpu torchaudio==2.5.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY model/ model/
COPY api/ api/

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
