FROM python:3.13-slim

# Slow or unstable connections: wait longer, retry more, and resume interrupted downloads.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10 \
    PIP_RESUME_RETRIES=10 \
    ENVIRONMENT=production

WORKDIR /app

# Install dependencies first so code changes do not invalidate this layer.
# Only the API's requirements: torch and matplotlib (training only) are in requirements-dev.txt.
# The cache mount keeps downloaded packages between builds, so a failed build resumes
# instead of downloading everything again. It is not part of the final image.
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

COPY . .

# Strip Windows line endings (CRLF breaks /bin/sh) and make the script executable.
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN sed -i 's/\r$//' /docker-entrypoint.sh && chmod +x /docker-entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/bin/sh", "/docker-entrypoint.sh"]
# main.py lives at the project root, so the app module is main:app.
# Request logs come from RequestLoggingMiddleware as JSON, so uvicorn's plain access log is off.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
