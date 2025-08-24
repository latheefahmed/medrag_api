# syntax=docker/dockerfile:1
FROM python:3.12-slim

# System deps you wanted
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential curl libopenblas-dev libgomp1 ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps (requirements.txt is in repo root)
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r /app/requirements.txt

# Copy app source from CURRENT repo root (not /medrag_api)
COPY app /app/app
COPY gunicorn_conf.py /app/gunicorn_conf.py
COPY run.py /app/run.py

# (optional) cache the embedder to speed first request
RUN python - <<'PY'
from sentence_transformers import SentenceTransformer
SentenceTransformer('pritamdeka/S-PubMedBERT-MS-MARCO')
print("Model cached.")
PY

# Runtime
ENV PYTHONUNBUFFERED=1 PORT=8000
EXPOSE 8000

# (optional) healthcheck; remove if /debug/health doesn't exist
# HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
#   CMD curl -fsS http://127.0.0.1:${PORT}/debug/health || exit 1

CMD ["gunicorn", "app.main:app", "-c", "gunicorn_conf.py"]
