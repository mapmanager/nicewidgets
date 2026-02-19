# Dockerfile for Plot Pool app (nicewidgets)
# Web mode: PLOT_POOL_GUI_NATIVE=0, bind to $HOST:$PORT
#
# Production build:
#   docker build -t nicewidgets-plot-pool:latest .
#
# Development build (editable install, picks up local src changes):
#   docker build --build-arg DEV_MODE=true -t nicewidgets-plot-pool:dev .
#
# Run container:
#   docker run --rm -p 8080:8080 nicewidgets-plot-pool:latest
#
# Run dev container with volume mount:
#   docker run --rm -p 8080:8080 -v $(pwd)/src:/app/src nicewidgets-plot-pool:dev

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy metadata first for caching
COPY pyproject.toml README.md LICENSE* ./

# Copy source
COPY src ./src

# Include sample CSV data (schema.get_data_dir() resolves to pkg_root/data)
COPY data ./data

# Build argument for editable install
ARG DEV_MODE=false

# Install (editable for both modes so schema.get_data_dir() resolves to /app/data)
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir -e ".[no_mpl]"

ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=8080

# Force web behavior in container
ENV PLOT_POOL_GUI_NATIVE=0
ENV PLOT_POOL_GUI_RELOAD=0

EXPOSE 8080

CMD ["python", "-m", "nicewidgets.plot_pool_app.plot_pool_app"]
