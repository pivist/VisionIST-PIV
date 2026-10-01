# syntax=docker/dockerfile:1
#
# VisionIST webui — the FastAPI app that serves the SPA at / and the /api/*
# control plane, proxying calls to the box fleet (reachable here by service
# name when run in the visionist-fleet compose network).
#
# Build from the REPO ROOT (the context must contain BOTH visionist_client/
# AND webui/):
#   docker build -f webui/Dockerfile -t visionist-local/webui .
#   # or, via compose:  cd fleet && docker compose build webui
#
# No prerequisites: the SPA is built inside the image (the `spa` stage below),
# so a fresh clone builds as-is — `webui/web/dist` is gitignored and is NOT
# expected in the context.  For local front-end work you can still run
# `(cd webui/web && npm ci && npm run build)`; the image ignores that output
# and always builds its own.
#
# Notes:
#   - webui is installed *editable* on purpose: app.py resolves webui/web/dist
#     relative to this source tree, so the SPA stays correctly located.
#   - No GPU needed here: the webui is a thin CPU front-end over the boxes.


# ----------------------------------------------------------------------------
# Stage 1 — build the SPA (React + Vite + TypeScript)
# ----------------------------------------------------------------------------
FROM node:20-slim AS spa

WORKDIR /app

# Lockfile first: `npm ci` is reproducible and this layer caches until the
# dependency set actually changes.
COPY webui/web/package.json webui/web/package-lock.json ./
RUN npm ci --no-audit --no-fund

# Sources (node_modules and build junk stay out via the repo-root .dockerignore)
COPY webui/web/ ./

# `npm run build` = `tsc --noEmit && vite build` -> /app/dist
RUN npm run build


# ----------------------------------------------------------------------------
# Stage 2 — the runtime image
# ----------------------------------------------------------------------------
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    WEBUI_HOST=0.0.0.0 \
    WEBUI_PORT=8000 \
    WEBUI_DATA_DIR=/data

WORKDIR /opt

# Only what the app needs at runtime (skip node_modules, tests, and build
# junk — see the repo-root .dockerignore).
COPY visionist_client/pyproject.toml      visionist_client/
COPY visionist_client/README.md           visionist_client/
COPY visionist_client/LICENSE             visionist_client/
COPY visionist_client/src/                visionist_client/src/
COPY webui/pyproject.toml                 webui/
COPY webui/README.md                      webui/
COPY webui/src/                           webui/src/
COPY webui/boxes/                         webui/boxes/
COPY --from=spa /app/dist/                webui/web/dist/

# Install the local client (satisfies webui's visionist-client dep, and keeps
# the exact same gRPC stubs as the local wheel), then the webui.
#
# torch is REQUIRED to decode torch.save() payloads (tapnext/vggt/tapnext
# tracks & visibles).  Without it the codec silently degrades to `identity`
# and every tensor arrives at the SPA as an opaque `application/zip` file —
# the tensor panel loses dtype/shape/size and its .npy download button, and
# the tracks viewer gets nothing to play.  Use PyTorch's CPU wheel index:
# webui is a CPU front-end and the CUDA build would add ~2 GB of nvidia libs.
RUN pip install --no-cache-dir --upgrade pip \
 &&  pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 &&  pip install --no-cache-dir ./visionist_client \
 &&  pip install --no-cache-dir -e ./webui

EXPOSE 8000
VOLUME ["/data"]

# Cheap liveness: the FastAPI service-info route (added before the SPA mount).
HEALTHCHECK --interval=15s --timeout=4s --start-period=20s \
  CMD python -c "import urllib.request as u;u.urlopen('http://127.0.0.1:8000/api/service',timeout=3)" || exit 1

CMD ["uvicorn", "webui.app:factory", "--host", "0.0.0.0", "--port", "8000"]
