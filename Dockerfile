# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps (same as before)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential curl libopenblas-dev libgomp1 ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python deps ────────────────────────────────────────────────────────────────
# We assume requirements.txt lives in medrag_api/ (same place as this Dockerfile)
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r /app/requirements.txt

# ── App code ──────────────────────────────────────────────────────────────────
# Your FastAPI app package is medrag_api/app/*
COPY app /app/app

# If you keep these files in medrag_api/, copy them (safe even if you don’t use run.py)
# If you don’t have gunicorn_conf.py yet, you can add the minimal one below.
COPY gunicorn_conf.py /app/gunicorn_conf.py
# COPY run.py /app/run.py  # uncomment if you actually have/run this

# ── (Optional) Pre-cache the embedding model via LlamaIndex ───────────────────
# This pulls pritamdeka/S-PubMedBERT-MS-MARCO once at build time so first request is fast.
RUN python - <<'PY'
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
m = HuggingFaceEmbedding(model_name="pritamdeka/S-PubMedBERT-MS-MARCO")
_ = m.get_text_embedding("cache please")
print("LlamaIndex HF embedding cached.")
PY

# Runtime
ENV PORT=8000
EXPOSE 8000

# If you prefer uvicorn directly, replace CMD with:
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Gunicorn with Uvicorn workers (keeps your previous style)
CMD ["gunicorn", "app.main:app", "-c", "gunicorn_conf.py"]
