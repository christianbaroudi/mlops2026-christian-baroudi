# The image of every job (preprocess, features, train, score) and of the service.
# Built and pushed to ghcr.io by CD on every merge to master.
FROM python:3.11-slim

# uv, copied from its official image.
COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /uvx /bin/

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

# Dependencies first: this layer is reused as long as uv.lock doesn't change.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

# Then our code.
COPY src ./src
COPY scripts ./scripts
RUN uv sync --locked --no-dev

# There is no .git in the image: CD passes the commit, and training logs it.
# GIT_PYTHON_REFRESH=quiet stops MLflow from warning that git is missing.
ARG GIT_COMMIT=unknown
ENV GIT_COMMIT=$GIT_COMMIT \
    GIT_PYTHON_REFRESH=quiet

# By default, the image starts the service. Compose gives each job its own command.
EXPOSE 8000
CMD ["python", "scripts/serve.py"]
