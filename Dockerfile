# Serving image: the agent, approval queue, MCP server, webhook, review app,
# and evals. The Spark/Delta pipeline stays out of it (no JVM, no pyspark);
# only the gold snapshot the agent reads is baked in. One image, three
# services: docker-compose.yml and infra/main.bicep run it with different
# commands (review app by default, MCP server, webhook).
#
#   docker build -t music-man .
#   docker run --rm -p 8501:8501 music-man
#   docker compose up            # all three services + Postgres
#   docker run --rm --env-file .env music-man \
#       python -m music_man.agent.run_agent --dry-run "Find the most suspicious artist"

FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /bin/uv

WORKDIR /app
COPY pyproject.toml ./
RUN uv pip install --system --no-cache -r pyproject.toml --extra agents --extra webhook --extra cloud

COPY src ./src
COPY evals ./evals
COPY data/gold/artist_daily_metrics.snapshot.parquet ./data/gold/

# Run from source so PROJECT_ROOT resolves to /app.
ENV PYTHONPATH=/app/src:/app \
    PYTHONUNBUFFERED=1

RUN useradd --create-home --uid 1000 app && chown -R app /app/data
USER app

EXPOSE 8501 8765 8800
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "src/music_man/review_app/app.py", "--server.address=0.0.0.0", "--server.headless=true"]
