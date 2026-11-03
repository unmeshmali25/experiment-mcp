FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src

RUN uv sync --frozen --no-dev --no-editable \
    && useradd --create-home --uid 10001 app \
    && mkdir -p /data \
    && chown app:app /data

ENV PATH="/app/.venv/bin:$PATH" \
    EXPERIMENT_MCP_DB=/data/experiments.db \
    EXPERIMENT_MCP_HOST=0.0.0.0 \
    EXPERIMENT_MCP_PORT=8000

USER app
EXPOSE 8000

# Auth is required. Set EXPERIMENT_MCP_ISSUER_URL, EXPERIMENT_MCP_RESOURCE_URL,
# and EXPERIMENT_MCP_JWKS_URL or EXPERIMENT_MCP_JWKS_PATH. The process exits
# if those are missing. EXPERIMENT_MCP_ALLOW_INSECURE_NO_AUTH=1 disables auth
# and is only for a local throwaway container.
CMD ["experiment-mcp", "serve", "--transport", "http", "--host", "0.0.0.0", "--port", "8000"]
