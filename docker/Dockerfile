# Stage 1: base image with CUDA + Python
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip git curl \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Stage 2: install Python dependencies
FROM base AS deps

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 3: final image
FROM deps AS final

COPY . .

ENV MODEL_PATH=/app/merged \
    MAX_NEW_TOKENS=512 \
    PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=4 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

CMD ["uvicorn", "serving.server:app", "--host", "0.0.0.0", "--port", "8000"]
