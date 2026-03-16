FROM python:3.12-slim

# System deps for geopandas/shapely/GDAL
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev \
    libgeos-dev \
    libproj-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install poetry
RUN pip install --no-cache-dir poetry==1.8.3

WORKDIR /app

# Copy dependency files first so this layer is cached unless deps change
COPY pyproject.toml poetry.lock ./

# Install dependencies into the system Python (no virtualenv needed in Docker)
RUN poetry config virtualenvs.create false \
    && poetry install --no-root --only main --no-interaction --no-ansi

# Copy source
COPY mobile_coverage/ mobile_coverage/

# Install the package itself
RUN poetry install --only main --no-interaction --no-ansi

# Results are written here — mount a host directory to persist output
RUN mkdir -p data

COPY entrypoint.sh ./
RUN chmod +x entrypoint.sh

# Pass your HuggingFace token via .env or -e HF_TOKEN=hf_...
ENV HF_TOKEN=""

CMD ["./entrypoint.sh"]
