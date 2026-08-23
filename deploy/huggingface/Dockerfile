# Hugging Face Space (Docker SDK) build of the SickNote API.
#
# Differs from the repo-root Dockerfile in two ways only:
#   - listens on 7860, the port HF Spaces expects by default
#   - runs as a non-root user, which Spaces requires for writes
#
# Everything else — CPU-only torch, ffmpeg, lazy model loading — is identical,
# so what runs here is the same service that runs on Railway.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Spaces runs containers as uid 1000 and mounts a writable /home/user.
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
WORKDIR /home/user/app

COPY --chown=user requirements.txt .

RUN pip install --no-cache-dir --user \
        torch torchaudio --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user model/ model/
COPY --chown=user api/ api/

EXPOSE 7860

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1 --timeout-keep-alive 30"]
