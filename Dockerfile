# Serving image: the agent, approval queue, review app, and evals.
# The Spark/Delta pipeline stays out of it (no JVM, no pyspark) - it
# produces the gold snapshot that gets mounted in at /app/data.
#
#   docker build -t music-man .
#   docker run --rm -p 8501:8501 -v ./data:/app/data music-man
#   docker run --rm --env-file .env -v ./data:/app/data music-man \
#       python -m music_man.agent.run_agent --dry-run "Find the most suspicious artist"

FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /bin/uv

WORKDIR /app
COPY pyproject.toml ./
RUN uv pip install --system --no-cache -r pyproject.toml --extra agents --extra webhook

COPY src ./src
COPY evals ./evals

# Run from source so PROJECT_ROOT resolves to /app and data/ is the mounted volume.
ENV PYTHONPATH=/app/src:/app \
    PYTHONUNBUFFERED=1

RUN useradd --create-home --uid 1000 app && mkdir -p /app/data && chown app /app/data
USER app

EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "src/music_man/review_app/app.py", "--server.address=0.0.0.0", "--server.headless=true"]
