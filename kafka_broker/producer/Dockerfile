FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY bluesky_producer.py .

# Create non-root user for security
RUN useradd -m -u 1000 producer && \
    chown -R producer:producer /app
USER producer

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Run the producer
CMD ["python", "-u", "bluesky_producer.py"]