# Container image for the poverty-targeting prediction API.
# Build:  docker build -t poverty-api:0.1.0 .
# Run:    docker run --rm -p 8080:8080 poverty-api:0.1.0

# Same Python version the model was trained with.
FROM python:3.13-slim

# LightGBM needs the OpenMP runtime (libgomp1); slim images leave it out.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install the package with only the API extra (no devecon, no training tools),
# using the exact library versions the model file was saved with.
COPY docker/constraints.txt ./constraints.txt
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --constraint constraints.txt ".[api]"

# The trained model, and where the app should look for it.
COPY models/GH2022DHS_tierB_lightgbm.joblib ./models/
ENV POVERTY_TARGETING_MODEL=/app/models/GH2022DHS_tierB_lightgbm.joblib

# Never run a web server as root.
RUN useradd --create-home --uid 10001 appuser
USER appuser

# Cloud Run tells the container which port to use through $PORT (8080 by default).
ENV PORT=8080
EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn poverty_targeting.api.app:create_app --factory --host 0.0.0.0 --port ${PORT}"]