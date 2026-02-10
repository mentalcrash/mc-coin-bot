# ==============================================================================
# MC Coin Bot — Multi-stage Docker build with uv
# ==============================================================================
# Stage 1: Build dependencies with uv
# Stage 2: Minimal runtime image
# ==============================================================================

# ---------------------------------------------------------------------------
# Stage 1: Builder
# ---------------------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# 1) 의존성 레이어 (pyproject.toml + uv.lock만 복사 → Docker cache 극대화)
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# 2) 소스 레이어
COPY src/ src/
COPY main.py ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ---------------------------------------------------------------------------
# Stage 2: Runtime
# ---------------------------------------------------------------------------
FROM python:3.13-slim-bookworm

# Non-root user
RUN groupadd --gid 1000 bot && \
    useradd --uid 1000 --gid bot --create-home bot

WORKDIR /app

# Builder에서 .venv만 복사
COPY --from=builder --chown=bot:bot /app/.venv /app/.venv
COPY --from=builder --chown=bot:bot /app/src /app/src
COPY --from=builder --chown=bot:bot /app/main.py /app/main.py

# Config & healthcheck 복사
COPY --chown=bot:bot config/ /app/config/
COPY --chown=bot:bot scripts/healthcheck.py /app/scripts/healthcheck.py

# 데이터/로그 디렉토리 (volume mount 대상)
RUN mkdir -p /app/data /app/logs && chown -R bot:bot /app/data /app/logs

# PATH에 .venv 추가
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=agg

EXPOSE 8000

USER bot

# Health check (HTTP 서버 없이 프로세스 상태 확인)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD ["python", "scripts/healthcheck.py"]

# ENTRYPOINT + CMD 분리 → 모드 오버라이드 가능
# 기본: mc-bot (pyproject.toml [project.scripts] 진입점)
ENTRYPOINT ["mc-bot"]
CMD []
