FROM python:3.11-slim

# Copy the uv binary directly from the official uv image — avoids pip entirely.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Unbuffered stdout/stderr so container logs appear in real time.
# UV_COMPILE_BYTECODE pre-compiles .pyc during install to speed up cold starts.
# UV_LINK_MODE=copy prevents hardlink failures when the uv cache is on a
# different filesystem than the project venv (common in CI/Docker).
ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Copy dependency manifests first so Docker can cache the install layer
# independently from source changes.
COPY pyproject.toml uv.lock README.md ./

# Install production dependencies into .venv using the exact locked versions.
# --no-dev skips the [dependency-groups] dev extras (pytest-cov, pytest-mock).
RUN uv sync --frozen --no-dev

# Put the venv on PATH so uvicorn and streamlit are callable directly.
ENV PATH="/app/.venv/bin:$PATH"

# Copy the rest of the application source after deps are installed so that
# source edits don't invalidate the expensive uv sync layer.
COPY . .

# FastAPI
EXPOSE 8000
# Streamlit
EXPOSE 8501

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
