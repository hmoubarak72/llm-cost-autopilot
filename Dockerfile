FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    AUTOPILOT_ROOT=/app AUTOPILOT_VAR_DIR=/var/autopilot AUTOPILOT_DB=/var/autopilot/autopilot.db \
    AUTOPILOT_CLASSIFIER=/var/autopilot/classifier.joblib AUTOPILOT_LOG=/var/autopilot/requests.jsonl
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[dashboard]"
COPY config ./config
COPY data ./data
COPY dashboard ./dashboard
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000 8501
ENTRYPOINT ["/entrypoint.sh"]
CMD ["serve"]
