FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential curl libopenblas-dev libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app

COPY medrag_api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY medrag_api /app

# (optional) cache the embedder to speed first request
RUN python - <<'PY'
from sentence_transformers import SentenceTransformer
SentenceTransformer('pritamdeka/S-PubMedBERT-MS-MARCO')
print("Model cached.")
PY

EXPOSE 8000
ENV PYTHONUNBUFFERED=1 PORT=8000
CMD ["gunicorn", "app.main:app", "-c", "gunicorn_conf.py"]
