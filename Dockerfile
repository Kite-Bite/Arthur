# syntax=docker/dockerfile:1
# Arthur runs as a local process that talks to an Ollama server. The image
# contains Arthur only - point OLLAMA_HOST (or ARTHUR_LLM_HOST) at wherever
# Ollama runs.
#
# Ollama binds 127.0.0.1:11434 by default, so a container cannot reach it
# over the bridge. On Linux the simplest fix is host networking:
#
#   docker build -t arthur .
#   docker run --rm --network host arthur
#
# Without host networking, add the gateway alias and expose Ollama:
#
#   docker run --rm -p 8420:8420 \
#     --add-host=host.docker.internal:host-gateway \
#     -e OLLAMA_HOST=http://host.docker.internal:11434 arthur
#
# Both stages use the official python:3.12-slim image so the virtualenv built
# in stage one resolves against the exact same interpreter in stage two.
# The build deliberately avoids BuildKit-only syntax (cache mounts) so it
# works with the legacy builder too.

FROM python:3.12-slim-bookworm AS build

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies first: this layer is cached until pyproject/uv.lock change.
# The uv cache lives outside /app, so it is discarded with this stage rather
# than shipped in the runtime image.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN uv sync --frozen --no-dev

# ---------------------------------------------------------------------------

FROM python:3.12-slim-bookworm AS runtime

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 --shell /usr/sbin/nologin arthur \
    # Pre-create the data dir: docker would otherwise create the declared
    # volume owned by root, and the unprivileged user could not write to it.
    && mkdir -p /home/arthur/.local/share/arthur /home/arthur/.config \
    && chown -R arthur:arthur /home/arthur

COPY --from=build /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Containers must bind the interface, not loopback.
    ARTHUR_API_HOST=0.0.0.0 \
    ARTHUR_API_PORT=8420

# State lives here; mount a volume to keep memory, documents and audit history.
VOLUME ["/home/arthur/.local/share/arthur"]
EXPOSE 8420

USER arthur
WORKDIR /home/arthur

# Liveness only: this checks that Arthur's HTTP surface answers, not that the
# model is reachable. Ollama runs outside this container, so failing liveness
# when it is down would restart Arthur to no benefit - /health still reports
# "degraded" for that. The port is read from the env so overriding
# ARTHUR_API_PORT keeps the probe working.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python3 -c "import os,urllib.request,sys; p=os.environ.get('ARTHUR_API_PORT','8420'); sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{p}/health', timeout=4).status == 200 else 1)"

ENTRYPOINT ["arthur"]
CMD ["serve"]
