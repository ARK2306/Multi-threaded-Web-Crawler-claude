# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    SEARCH_ENGINE_DB_PATH=/data/search.db

WORKDIR /app

# Dependencies first so code edits do not invalidate the pip layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py storage.py crawler.py indexer.py api.py cli.py ./

# Run as a non-root user. /data is created here so that the named volume mounted
# over it inherits this ownership and the app can write the SQLite file.
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /data \
    && chown -R app:app /data /app
USER app

# The config is expected at ~/.secrets/search_engine_config.json; docker-compose
# mounts the host's ~/.secrets here read-only.
VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"

ENTRYPOINT ["python", "cli.py"]
CMD ["serve"]
