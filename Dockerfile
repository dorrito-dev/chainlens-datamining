# ChainLens Phase 1 application image.
# Python 3.11 + uv-managed dependencies; the repository is bind-mounted at /app.
FROM python:3.11-slim

ARG UV_VERSION=0.8.22
ARG INSTALL_BIGQUERY=0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    UV_SYSTEM_PYTHON=1 \
    UV_HTTP_TIMEOUT=300

WORKDIR /app

RUN pip install --no-cache-dir "uv==${UV_VERSION}"

# Dependencies first so rebuilds only reinstall when requirements change.
COPY requirements.txt requirements-bigquery.txt pyproject.toml ./
RUN uv pip install --system -r requirements.txt \
    && if [ "${INSTALL_BIGQUERY}" = "1" ]; then uv pip install --system -r requirements-bigquery.txt; fi

COPY . .

CMD ["sleep", "infinity"]
