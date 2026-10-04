# syntax=docker/dockerfile:1
# Arthur runs as a local process that talks to an Ollama server. The image
# contains Arthur only - point OLLAMA_HOST (or ARTHUR_LLM_HOST) at wherever
# Ollama runs, e.g. the host: http://host.docker.internal:11434
#
#   docker build -t arthur .
#   docker run --rm -p 8420:8420 -e OLLAMA_HOST=http://host.docker.internal:11434 arthur
#
# Both stages use the official python:3.12-slim image so the virtualenv built
# in stage one resolves against the exact same interpreter in stage two.

FROM python:3.12-slim-bookworm AS build

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies first: this layer is cached until pyproject/uv.lock change.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

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

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python3 -c "import urllib.request,sys; \
        sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8420/health', timeout=4).status == 200 else 1)"

ENTRYPOINT ["arthur"]
CMD ["serve"]
