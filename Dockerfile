# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1


RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential curl libopenblas-dev libgomp1 ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r /app/requirements.txt

COPY app /app/app
COPY gunicorn_conf.py /app/gunicorn_conf.py

RUN python - <<'PY'
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
m = HuggingFaceEmbedding(model_name="pritamdeka/S-PubMedBERT-MS-MARCO")
_ = m.get_text_embedding("cache please")
print("LlamaIndex HF embedding cached.")
PY


ENV PORT=8000
EXPOSE 8000

CMD ["gunicorn", "app.main:app", "-c", "gunicorn_conf.py"]
