FROM python:3.12-slim

# Security: run as non-root
RUN groupadd -r bot && useradd -r -g bot bot

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Create necessary directories
RUN mkdir -p logs state && chown -R bot:bot /app

USER bot

# Health check via metrics endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:9090/health')" || exit 1

EXPOSE 9090

CMD ["python", "-m", "bot"]
