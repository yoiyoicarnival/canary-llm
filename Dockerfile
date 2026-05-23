FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir ".[server]" && \
    python -c "from canary.scorer import build_bank; build_bank()"

EXPOSE 8080
CMD ["uvicorn", "canary.server:app", "--host", "0.0.0.0", "--port", "8080"]
