# 1. Base Image: Slim Python 3.11 for low vulnerability surface and lean footprint
FROM python:3.11-slim

# 2. Prevent Python from buffering stdout/stderr and writing .pyc files
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=5000

# 3. Set Working Directory inside the container
WORKDIR /app

# 4. Install OS-level dependencies (if needed by Pillow)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 5. Install Python dependencies first (Layer caching optimization)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy Application Code, Model, and Assets
COPY app/ ./app/

# 7. Create directory for persistent runtime storage (SQLite DB & Logs)
RUN mkdir -p /app/reports && chmod -R 777 /app/reports

# 8. Create and switch to non-root user for security hardening
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# 9. Expose application port
EXPOSE 5000

# 10. Healthcheck instruction to ensure container self-monitoring
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# 11. Production WSGI Entrypoint via Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "--timeout", "120", "app.app:app"]